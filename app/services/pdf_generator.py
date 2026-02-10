"""
PDF 生成模块

将 Markdown 总结转换为 PDF (WeasyPrint 渲染)。
支持 LaTeX 公式 (matplotlib 转图片) 和 Markdown 格式修正。
"""
import logging
import re
import base64
import io
import os
import tempfile
import gc

logger = logging.getLogger(__name__)


# ======================== Cleanup ========================

def cleanup_ai_output(content: str) -> str:
    """清理 AI 输出 (移除思维链、搜索标签等)"""
    # 移除 <search>...</search> 块
    content = re.sub(r'<search>.*?</search>\s*', '', content, flags=re.DOTALL)
    
    # 移除 <query>...</query> 标签
    content = re.sub(r'<query>.*?</query>\s*', '', content, flags=re.DOTALL)
    
    # 移除开头的空行
    content = content.lstrip('\n')

    # 截取到第一个一级标题 (去除思维链/工具日志)
    # 查找第一个以 "# " 开头的行
    match = re.search(r'^#\s+.+', content, flags=re.MULTILINE)
    if match:
        # 保留从标题开始的内容
        content = content[match.start():]
    
    return content


# ======================== LaTeX Renderer ========================

def render_latex_to_base64(latex_code: str, fontsize: int = 12, dpi: int = 72) -> str:
    """使用 matplotlib 将 LaTeX 公式渲染为 base64 PNG"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt

        # 配置字体以支持中文
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Noto Sans CJK JP'
        plt.rcParams['mathtext.it'] = 'Noto Sans CJK JP:italic'
        plt.rcParams['mathtext.bf'] = 'Noto Sans CJK JP:bold'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False     

        latex = _preprocess_latex(latex_code)
        
        fig = Figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0)
        FigureCanvasAgg(fig)

        # 尝试渲染
        text = fig.text(0, 0, f"${latex}$", fontsize=fontsize, usetex=False)
        fig.canvas.draw()
        bbox = text.get_window_extent()

        width = bbox.width / fig.dpi + 0.1
        height = bbox.height / fig.dpi + 0.1
        fig.set_size_inches(width, height)
        text.set_position((0.05, 0.2))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02, transparent=True)
        gc.collect()

        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        logger.warning(f"LaTeX 渲染失败: {e}")
        return None


def _preprocess_latex(latex: str) -> str:
    """预处理 LaTeX 公式"""
    # \text{} -> \mathrm{} (matplotlib 不支持 \text)
    return re.sub(r'\\text\{([^}]*)\}', r'\\mathrm{\1}', latex)


# ======================== LaTeX processing in Markdown ========================

def _fix_blockquote_latex(content: str) -> str:
    """提取引用块中的 LaTeX 公式，避免正则匹配失败"""
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        if line.strip().startswith('>'):
            match = re.match(r'^(\s*>\s*)', line)
            if match:
                prefix = match.group(1)
                rest = line[len(prefix):]
                
                # 处理块级 LaTeX: $$...$$
                if '$$' in rest:
                    rest = re.sub(
                        r'\$\$(.+?)\$\$',
                        lambda m: _render_latex_inline(m.group(1), block=True),
                        rest,
                        flags=re.DOTALL
                    )
                
                # 处理行内 LaTeX: $...$
                if '$' in rest:
                    rest = re.sub(
                        r'(?<!\$)\$([^\$\n]+?)\$(?!\$)',
                        lambda m: _render_latex_inline(m.group(1), block=False),
                        rest
                    )
                
                line = prefix + rest
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def _render_latex_inline(latex_code: str, block: bool = False) -> str:
    """渲染单个 LaTeX 公式并返回 HTML img 标签"""
    latex = latex_code.strip()
    if not latex:
        return f'$${latex_code}$$' if block else f'${latex_code}$'
    
    b64 = render_latex_to_base64(latex, fontsize=10, dpi=72)
    if b64:
        if block:
            return (f'<img src="data:image/png;base64,{b64}" '
                    f'alt="formula" class="block-formula" style="display:block;margin:0 auto;"/>')
        else:
            return (f'<img src="data:image/png;base64,{b64}" '
                    f'alt="formula" class="inline-formula"/>')
    else:
        return f'<code>{latex}</code>'


def process_latex_in_markdown(content: str) -> str:
    """查找 Markdown 中的 LaTeX 公式并替换为图片"""
    def replace_block_latex(match):
        latex = match.group(1).strip()
        b64 = render_latex_to_base64(latex, fontsize=10, dpi=72)
        if b64:
            return (f'\n\n<p style="text-align:center;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'alt="formula" class="block-formula"/></p>\n\n')
        return f'\n\n<pre><code>{latex}</code></pre>\n\n'

    def replace_math_block(match):
        return replace_block_latex(match)

    def replace_inline_latex(match):
        latex = match.group(1).strip()
        if not latex: return match.group(0)
        b64 = render_latex_to_base64(latex, fontsize=10, dpi=72)
        if b64:
            return (f'<img src="data:image/png;base64,{b64}" '
                    f'alt="formula" class="inline-formula"/>')
        return f'<code>{latex}</code>'

    content = re.sub(r'\$\$(.+?)\$\$', replace_block_latex, content, flags=re.DOTALL)
    content = re.sub(r'```math\s*\n(.+?)\n```', replace_math_block, content, flags=re.DOTALL)
    content = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', replace_inline_latex, content)

    return content


# ======================== Markdown Normalizer ========================

_CONTENT_END_RE = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\uff01-\uff60\u3000-\u303fa-zA-Z)\]>»）】》\u300b\'""\u201d\u300d\.!?\?？。！%‰]' 
)
_UL_MARKER_RE = re.compile(r'^[*\-+]([ \t]+)')
_OL_MARKER_RE = re.compile(r'^\d{1,3}\.([ \t]+)')


def _match_list_marker(text: str):
    m = _UL_MARKER_RE.match(text)
    if m: return m
    return _OL_MARKER_RE.match(text)


def _paren_depth_at(text: str, pos: int) -> int:
    """计算括号深度"""
    depth = 0
    for i in range(pos):
        c = text[i]
        if c in '(（[【': depth += 1
        elif c in ')）]】': depth = max(0, depth - 1)
    return depth


def _split_inline_list_items(line: str) -> str:
    """拆分单行内的多个列表项"""
    if not line or line.startswith('#'): return line

    parts = []
    remaining = line

    head_marker = _match_list_marker(remaining)
    scan_start = head_marker.end() if head_marker else 0

    while scan_start < len(remaining):
        match = re.search(r'([ \t]+)(?=[*\-+][ \t]|\d{1,3}\.[ \t])', remaining[scan_start:])
        if not match: break

        ws_start = scan_start + match.start()
        marker_start = scan_start + match.end()

        if (ws_start > 0 and _CONTENT_END_RE.match(remaining[ws_start - 1]) 
                and _paren_depth_at(remaining, ws_start) == 0):
            marker_match = _match_list_marker(remaining[marker_start:])
            if marker_match:
                parts.append(remaining[:ws_start])
                remaining = remaining[marker_start:]
                head_marker = _match_list_marker(remaining)
                scan_start = head_marker.end() if head_marker else 0
                continue

        scan_start = marker_start + 1

    parts.append(remaining)
    return '\n\n'.join(parts)


def _fix_colon_then_list(line: str) -> str:
    """处理 '文字：* 列表'"""
    m = re.match(r'^(.*[：:])(\s*)([*\-+]|\d{1,3}\.)([ \t]+.+)$', line)
    if m: return m.group(1) + '\n\n' + m.group(3) + m.group(4)
    return line


def _fix_marker_spacing(line: str) -> str:
    """修复列表标记缺空格"""
    if re.match(r'^[*\-+][^\s]', line): return line[0] + ' ' + line[1:]
    return line


def normalize_markdown(content: str) -> str:
    """鲁棒的 Markdown 格式修正"""
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # 保护 LaTeX
    latex_store = []
    def _stash_latex(match):
        idx = len(latex_store)
        latex_store.append(match.group(0))
        return f'\x00LATEX{idx}\x00'

    content = re.sub(r'\$\$.+?\$\$', _stash_latex, content, flags=re.DOTALL)
    content = re.sub(r'(?<!\$)\$[^\$\n]+?\$(?!\$)', _stash_latex, content)
    content = re.sub(r'```math\s*\n.+?\n```', _stash_latex, content, flags=re.DOTALL)

    output_lines = []
    in_code_block = False

    for line in content.split('\n'):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            output_lines.append(line)
            continue

        if in_code_block:
            output_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith(('#', '>', '---', '===')):
            output_lines.append(line)
            continue

        line = _fix_colon_then_list(line)
        
        # 拆分行内列表
        expanded = []
        for sl in line.split('\n'):
             expanded.append(_split_inline_list_items(sl))
        line = '\n'.join(expanded)

        # 修复空格
        fixed = []
        for sl in line.split('\n'):
            fixed.append(_fix_marker_spacing(sl))
        line = '\n'.join(fixed)

        output_lines.append(line)

    result = '\n'.join(output_lines)

    # 恢复 LaTeX
    for idx, original in enumerate(latex_store):
        result = result.replace(f'\x00LATEX{idx}\x00', original)

    return _ensure_blank_before_list(result)


def _ensure_blank_before_list(content: str) -> str:
    """确保列表项前有空行"""
    lines = content.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        result.append(line)
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()
            curr_strip = line.strip()
            
            is_next_list = bool(re.match(r'^[*\-+]\s+', next_line) or re.match(r'^\d+\.\s+', next_line))
            
            if is_next_list and curr_strip:
                if (curr_strip.endswith(('：', ':')) or
                    (not curr_strip.startswith(('#', '>', '-', '*', '+')) and
                     not re.match(r'^\d+\.', curr_strip))):
                    result.append('')
    
    return '\n'.join(result)


# ======================== PDF Generation ========================

GITHUB_PDF_CSS = """
@page { margin: 20mm 15mm; }
body {
    font-family: 'Noto Sans CJK SC', 'SimHei', sans-serif;
    font-size: 14px; line-height: 1.6; color: #1f2328;
    max-width: 980px; margin: 0 auto; padding: 45px;
    border: 1px solid #d1d9e0; border-radius: 6px;
}
h1, h2, h3 { margin-top: 24px; margin-bottom: 16px; font-weight: 600; color: #1f2328; }
h1 { font-size: 2em; border-bottom: 1px solid #d1d9e0; padding-bottom: 0.3em; }
h2 { font-size: 1.5em; border-bottom: 1px solid #d1d9e0; padding-bottom: 0.3em; }
p { margin-bottom: 16px; }
blockquote { color: #656d76; border-left: 0.25em solid #d1d9e0; padding: 0 1em; }
code { background-color: rgba(175, 184, 193, 0.2); padding: 0.2em 0.4em; border-radius: 6px; font-family: monospace; }
pre { background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 6px; }
table { border-collapse: collapse; margin-bottom: 16px; width: 100%; }
th, td { padding: 6px 13px; border: 1px solid #d1d9e0; }
tr:nth-child(2n) { background-color: #f6f8fa; }
img { max-width: 100%; }
img.block-formula { display: block; margin: 0 auto; }
.inline-formula { vertical-align: middle; }
"""

def generate_pdf(markdown_content: str, output_path: str) -> bool:
    """生成 PDF"""
    try:
        import markdown
        from weasyprint import HTML, CSS

        content = cleanup_ai_output(markdown_content)
        content = _fix_blockquote_latex(content)
        content = process_latex_in_markdown(content)
        content = normalize_markdown(content)

        html_content = markdown.markdown(
            content,
            extensions=['extra', 'tables', 'fenced_code', 'sane_lists'],
        )

        full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>文档</title></head>
<body class="markdown-body">{html_content}</body>
</html>"""

        html = HTML(string=full_html)
        css = CSS(string=GITHUB_PDF_CSS)
        html.write_pdf(output_path, stylesheets=[css])

        logger.info(f"PDF 生成成功: {output_path}")
        return True

    except Exception as e:
        logger.error(f"PDF 生成失败: {e}")
        return False
