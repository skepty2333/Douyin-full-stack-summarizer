#!/bin/bash
echo "=============================="
echo "API 连通性测试"
echo "=============================="

echo ""
echo "--- 1. 主站 Gemini (gemini-3.1-pro-preview-thinking) ---"
RESP=$(curl -s --connect-timeout 15 --max-time 60 https://sg.uiuiapi.com/v1/chat/completions \
  -H "Authorization: Bearer sk-6W2nsjT59X2Oy1foej3kCGolfJpVxW2iBnIpCB5NfMxDJ1m2" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.1-pro-preview-thinking","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "--- 2. 主站 Sonnet (claude-sonnet-4-6-thinking) ---"
RESP=$(curl -s --connect-timeout 15 --max-time 60 https://sg.uiuiapi.com/v1/chat/completions \
  -H "Authorization: Bearer sk-fKodAg2CVJi4DzAbJjwA13eKM3CFME5SY3d3t9lhA9S1hquK" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6-thinking","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "--- 3. 副站 Gemini (gemini-3.1-pro-preview-thinking) ---"
RESP=$(curl -s --connect-timeout 15 --max-time 60 https://api1.uiuiapi.com/v1/chat/completions \
  -H "Authorization: Bearer sk-IPs0k3GseTdVky26rYIFw5OE4dfBM3l4AK7lpbAtHWuz1rpS" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.1-pro-preview-thinking","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "--- 4. 副站 Sonnet (claude-sonnet-4-6-thinking) ---"
RESP=$(curl -s --connect-timeout 15 --max-time 60 https://api1.uiuiapi.com/v1/chat/completions \
  -H "Authorization: Bearer sk-icOMSLMHqj9CR9i36xp701cjYyiqoJsb8No8jhtpaNoKSXeT" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6-thinking","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "--- 5. Qwen (DashScope) ---"
RESP=$(curl -s --connect-timeout 15 --max-time 30 https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer sk-9e2bc05b4e1541e999f6eb4caf44be48" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "--- 6. DeepSeek ---"
RESP=$(curl -s --connect-timeout 15 --max-time 30 https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-b57c632eec8a4aecae1cdb28d232497b" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"reply OK"}],"max_tokens":5}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ✅', d['choices'][0]['message']['content']) if 'choices' in d else print('  ❌', d.get('error',{}).get('message', d))"

echo ""
echo "=============================="
echo "测试完成"
echo "=============================="
