"""
企业微信消息加解密模块
基于官方 WXBizMsgCrypt3.py 重写，适配 Python 3.10+
"""
import base64
import hashlib
import random
import socket
import string
import struct
import time
import xml.etree.cElementTree as ET

from Crypto.Cipher import AES


class WXBizMsgCryptError(Exception):
    """加解密错误"""
    ERROR_CODES = {
        -40001: "签名验证错误",
        -40002: "xml解析失败",
        -40003: "sha加密生成签名失败",
        -40004: "AESKey 非法",
        -40005: "ReceiveId 校验错误",
        -40006: "AES 加密失败",
        -40007: "AES 解密失败",
        -40008: "解密后得到的buffer非法",
        -40009: "base64加密失败",
        -40010: "base64解密失败",
        -40011: "生成xml失败",
    }

    def __init__(self, code):
        self.code = code
        self.message = self.ERROR_CODES.get(code, f"未知错误: {code}")
        super().__init__(self.message)


class PKCS7Encoder:
    """PKCS7 填充/去填充"""
    block_size = 32

    @classmethod
    def encode(cls, text: bytes) -> bytes:
        amount = cls.block_size - (len(text) % cls.block_size)
        if amount == 0:
            amount = cls.block_size
        return text + bytes([amount]) * amount

    @classmethod
    def decode(cls, decrypted: bytes) -> bytes:
        pad = decrypted[-1]
        if pad < 1 or pad > cls.block_size:
            pad = 0
        return decrypted[:-pad]


class WXBizMsgCrypt:
    """企业微信消息加解密"""

    def __init__(self, token: str, encoding_aes_key: str, receive_id: str):
        self.token = token
        self.receive_id = receive_id
        try:
            self.aes_key = base64.b64decode(encoding_aes_key + "=")
            assert len(self.aes_key) == 32
        except Exception:
            raise WXBizMsgCryptError(-40004)

    def _get_sha1(self, token, timestamp, nonce, encrypt) -> str:
        """SHA1签名"""
        try:
            sort_list = sorted([token, timestamp, nonce, encrypt])
            sha = hashlib.sha1("".join(sort_list).encode("utf-8"))
            return sha.hexdigest()
        except Exception:
            raise WXBizMsgCryptError(-40003)

    def _encrypt(self, text: str) -> str:
        """AES加密"""
        try:
            text_bytes = text.encode("utf-8")
            # 16字节随机字符串 + 4字节网络字节序长度 + 明文 + receive_id
            rand_bytes = ''.join(random.choices(string.ascii_letters + string.digits, k=16)).encode()
            length_bytes = struct.pack("I", socket.htonl(len(text_bytes)))
            receive_id_bytes = self.receive_id.encode("utf-8")

            plain = rand_bytes + length_bytes + text_bytes + receive_id_bytes
            padded = PKCS7Encoder.encode(plain)

            cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
            encrypted = cipher.encrypt(padded)
            return base64.b64encode(encrypted).decode("utf-8")
        except Exception:
            raise WXBizMsgCryptError(-40006)

    def _decrypt(self, encrypted: str) -> str:
        """AES解密"""
        try:
            cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
            decrypted = cipher.decrypt(base64.b64decode(encrypted))
            plain = PKCS7Encoder.decode(decrypted)

            content = plain[16:]  # 去掉16字节随机字符串
            xml_len = socket.ntohl(struct.unpack("I", content[:4])[0])
            xml_content = content[4:xml_len + 4].decode("utf-8")
            from_receive_id = content[xml_len + 4:].decode("utf-8")

            if from_receive_id != self.receive_id:
                raise WXBizMsgCryptError(-40005)

            return xml_content
        except WXBizMsgCryptError:
            raise
        except Exception:
            raise WXBizMsgCryptError(-40007)

    def verify_url(self, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> str:
        """
        验证URL有效性 (GET请求回调验证)
        返回解密后的echostr明文
        """
        signature = self._get_sha1(self.token, timestamp, nonce, echostr)
        if signature != msg_signature:
            raise WXBizMsgCryptError(-40001)
        return self._decrypt(echostr)

    def decrypt_msg(self, post_data: str, msg_signature: str, timestamp: str, nonce: str) -> str:
        """
        解密消息 (POST请求)
        返回解密后的XML明文
        """
        try:
            xml_tree = ET.fromstring(post_data)
            encrypt = xml_tree.find("Encrypt").text
        except Exception:
            raise WXBizMsgCryptError(-40002)

        signature = self._get_sha1(self.token, timestamp, nonce, encrypt)
        if signature != msg_signature:
            raise WXBizMsgCryptError(-40001)

        return self._decrypt(encrypt)

    def encrypt_msg(self, reply_msg: str, nonce: str, timestamp: str = None) -> str:
        """
        加密回复消息
        返回加密后的XML字符串
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        encrypt = self._encrypt(reply_msg)
        signature = self._get_sha1(self.token, timestamp, nonce, encrypt)

        resp_xml = f"""<xml>
<Encrypt><![CDATA[{encrypt}]]></Encrypt>
<MsgSignature><![CDATA[{signature}]]></MsgSignature>
<TimeStamp>{timestamp}</TimeStamp>
<Nonce><![CDATA[{nonce}]]></Nonce>
</xml>"""
        return resp_xml
