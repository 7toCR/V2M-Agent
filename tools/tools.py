import json
import re, ast
import logging
from typing import Optional, List, Dict, Any, Tuple

from tools.live_status import LiveStatus


def extract_thoughts_from_text(text: str) -> List[str]:
    thoughts = []

    try:

        cleaned_text = text.strip()
        if cleaned_text.startswith('```') and cleaned_text.endswith('```'):

            lines = cleaned_text.split('\n')
            if len(lines) >= 3:
                cleaned_text = '\n'.join(lines[1:-1])


        json_data = json.loads(cleaned_text)
        if isinstance(json_data, dict) and 'Thought' in json_data:
            thoughts.append(str(json_data['Thought']).strip())
            return thoughts
    except (json.JSONDecodeError, ValueError, TypeError):
        pass


    pattern = r'Thought:\s*(.*?)(?=\s*Thought:|\s*$)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    thoughts = [match.strip() for match in matches]  # 清理空白字符

    return thoughts


def extract_result_from_tools(_response: Any, _string: str) -> list:
    _result = _response.get(_string, [])
    if _result:
        if isinstance(_result, list) and len(_result) > 0:

            _first_element = _result[0]
            if hasattr(_first_element, "content") or isinstance(_first_element, dict):

                _last_result = _result[-1]
                if hasattr(_last_result, "content"):
                    _final_answer = _last_result.content
                elif isinstance(_last_result, dict):
                    _final_answer = _last_result.get("content", str(_last_result))
                else:
                    _final_answer = str(_last_result)
                if isinstance(_final_answer, str) and _final_answer.startswith('['):
                    try:
                        import ast
                        _final_answer = ast.literal_eval(_final_answer)
                    except (ValueError, SyntaxError):
                        pass
            else:
                _final_answer = _result
        else:
            _final_answer = _result
    else:
        _final_answer = []
    return _final_answer


def extract_actions_from_text(text: str) -> List[Tuple[str, Optional[Any]]]:

    actions = []

    try:

        cleaned_text = text.strip()
        if cleaned_text.startswith('```') and cleaned_text.endswith('```'):

            lines = cleaned_text.split('\n')
            if len(lines) >= 3:
                cleaned_text = '\n'.join(lines[1:-1])


        json_data = json.loads(cleaned_text)
        if isinstance(json_data, dict) and 'Action' in json_data:
            action_str = str(json_data['Action']).strip()

            if action_str.lower() == 'none':
                actions.append(('none', None))
            else:
                # 简单的action解析，假设格式为 "action_name" 或 "action_name param"
                parts = action_str.split(None, 1)  # 按空格分割，最多分成2部分
                action_name = parts[0]
                parameters = parts[1] if len(parts) > 1 else None
                actions.append((action_name, parameters))
            return actions
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 如果JSON解析失败，回退到传统正则表达式方法
    # 支持 "Action: name, 参数: {...}"
    pattern_with_param = r'Action:\s*(\w+)\s*,\s*参数\s*:\s*(.*?)(?=\s*Action:|\s*$)'
    for action_name, param_str in re.findall(pattern_with_param, text, re.IGNORECASE | re.DOTALL):
        param_str = param_str.strip()
        parameters = None
        if param_str and param_str.lower() != 'none':
            if param_str.startswith('{') and param_str.endswith('}'):
                try:
                    parameters = json.loads(param_str)
                except json.JSONDecodeError:
                    parameters = param_str
            else:
                parameters = param_str
        actions.append((action_name, parameters))

    # 支持 "Action: name {json}" 或 "Action: name" 简单格式
    if not actions:
        simple_match = re.search(r'Action:\s*(\w+)(?:\s+(.*))?', text, re.IGNORECASE | re.DOTALL)
        if simple_match:
            action_name = simple_match.group(1)
            param_str = (simple_match.group(2) or "").strip()
            parameters = None
            if param_str and param_str.lower() != 'none':
                if param_str.startswith('{') and param_str.endswith('}'):
                    try:
                        parameters = json.loads(param_str)
                    except json.JSONDecodeError:
                        parameters = param_str
                else:
                    parameters = param_str
            actions.append((action_name, parameters))

    return actions


def extract_from_json_format(text: str, key: str) -> str:
    """
    Extract value from JSON format response like {'Thought': 'content'} or {'Action': 'content'}

    Args:
        text: 包含JSON格式的文本
        key: 要提取的键名

    Returns:
        提取到的值，如果提取失败则返回空字符串
    """
    # Remove any leading/trailing whitespace
    text = text.strip()

    # Try to parse as JSON directly
    try:
        json_data = json.loads(text)
        if isinstance(json_data, dict) and key in json_data:
            value = json_data[key]
            return str(value) if value is not None else ""
    except (json.JSONDecodeError, ValueError):
        pass

    # 尝试提取JSON对象（可能包含在文本中）
    # 使用更智能的方法：从第一个 { 开始，找到匹配的 }
    try:
        start_pos = text.find('{')
        if start_pos != -1:
            brace_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if brace_count == 0 and end_pos > start_pos:
                json_str = text[start_pos:end_pos]
                try:
                    json_data = json.loads(json_str)
                    if isinstance(json_data, dict) and key in json_data:
                        value = json_data[key]
                        return str(value) if value is not None else ""
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    except Exception:
        pass

    # Try to extract from JSON-like string with regex (支持多行和转义字符)
    # 单引号格式（支持转义的单引号）
    pattern = rf"'{key}'\s*:\s*'((?:[^'\\]|\\.)*)'"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 处理转义字符
        result = match.group(1).replace("\\'", "'").replace("\\\\", "\\")
        return result

    # 双引号格式（支持转义的双引号和多行）
    pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 处理转义字符
        result = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        return result

    return ""


def extract_field_from_response(response: Any, field_name: str) -> Any:
    """
    从response对象中提取指定字段的内容，支持JSON格式和传统文本格式。
    这是一个通用函数，可以提取任意字段（如 "Result" 等）。

    根据prompt.py的响应模板，所有步骤都使用 {"Result": [...]} 格式：
    - Observation步骤: {"Result": ["结果1", "结果2", ...]}

    Args:
        response: LangChain响应对象或字符串
        field_name: 要提取的字段名（如 "Result"）

    Returns:
        提取到的值：
        - 对于 "Result": 返回列表（如果是数组）或字符串
        - 如果提取失败，返回空字符串或空列表
    """
    # 从response中提取文本内容
    if hasattr(response, "content"):
        text_content = response.content
    else:
        text_content = str(response)

    if not text_content:
        return "" if field_name != "Result" else []

    # 清理文本，移除可能的markdown代码块标记
    cleaned_text = text_content.strip()

    # 处理markdown代码块（支持 ```json 或 ``` 开头）
    if cleaned_text.startswith('```'):
        lines = cleaned_text.split('\n')
        if len(lines) >= 2:
            # 跳过第一行（可能是 ```json 或 ```）
            start_idx = 1 if lines[0].strip().startswith('```') else 0
            # 跳过最后一行（应该是 ```）
            end_idx = len(lines) - 1 if (lines[-1].strip() == '```' or lines[-1].strip().startswith('```')) else len(
                lines)
            if end_idx > start_idx:
                cleaned_text = '\n'.join(lines[start_idx:end_idx]).strip()

    # 首先尝试解析JSON格式
    try:
        # 尝试直接解析JSON
        json_data = json.loads(cleaned_text)
        if isinstance(json_data, dict):
            # 如果请求的字段存在，直接返回
            if field_name in json_data:
                value = json_data[field_name]
                # 对于Result字段，如果是列表，直接返回
                if field_name == "Result" and isinstance(value, list):
                    return value
                # 对于其他字段，返回字符串
                return str(value).strip() if value is not None else ""

            # 根据prompt.py模板，所有响应都使用Result字段
            # 如果请求Thought或Action，从Result数组中提取
            if "Result" in json_data and isinstance(json_data["Result"], list):
                result_list = json_data["Result"]
                if field_name == "Thought":
                    # Thought步骤：返回Result数组的第一个元素作为思考内容
                    if len(result_list) > 0:
                        thought = str(result_list[0]).strip()
                        return thought if thought.lower() != "none" else "None"
                    return ""
                elif field_name == "Action":
                    # Action步骤：返回Result数组的第一个元素作为工具名
                    if len(result_list) > 0:
                        action = str(result_list[0]).strip()
                        return action if action.lower() != "none" else "none"
                    return ""
                elif field_name == "Result":
                    # 直接返回Result数组
                    return result_list
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 尝试提取JSON对象（可能包含在文本中）
    # 使用更智能的方法：从第一个 { 开始，找到匹配的 }
    try:
        start_pos = cleaned_text.find('{')
        if start_pos != -1:
            brace_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(cleaned_text)):
                if cleaned_text[i] == '{':
                    brace_count += 1
                elif cleaned_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if brace_count == 0 and end_pos > start_pos:
                json_str = cleaned_text[start_pos:end_pos]
                try:
                    json_data = json.loads(json_str)
                    if isinstance(json_data, dict):
                        # 如果请求的字段存在，直接返回
                        if field_name in json_data:
                            value = json_data[field_name]
                            if field_name == "Result" and isinstance(value, list):
                                return value
                            return str(value).strip() if value is not None else ""

                        # 根据prompt.py模板，从Result字段提取
                        if "Result" in json_data and isinstance(json_data["Result"], list):
                            result_list = json_data["Result"]
                            if field_name == "Thought":
                                if len(result_list) > 0:
                                    thought = str(result_list[0]).strip()
                                    return thought if thought.lower() != "none" else "None"
                                return ""
                            elif field_name == "Action":
                                if len(result_list) > 0:
                                    action = str(result_list[0]).strip()
                                    return action if action.lower() != "none" else "none"
                                return ""
                            elif field_name == "Result":
                                return result_list
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
    except Exception:
        pass

    # 如果JSON解析失败，回退到正则表达式方法
    # 首先尝试提取Result字段（因为所有响应都使用Result字段）
    result_patterns = [
        r'"Result"\s*:\s*(\[[^\]]*\])',  # "Result": [...]
        r"'Result'\s*:\s*(\[[^\]]*\])",  # 'Result': [...]
        r'"Result"\s*:\s*\[(.*?)\]',  # "Result": [...] (支持多行)
        r"'Result'\s*:\s*\[(.*?)\]",  # 'Result': [...] (支持多行)
    ]

    for pattern in result_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            try:
                # 如果匹配到完整的数组
                if match.lastindex == 1 and match.group(1).startswith('['):
                    array_str = match.group(1)
                else:
                    # 如果只匹配到数组内容，需要加上括号
                    array_str = "[" + match.group(1) + "]"
                array_data = json.loads(array_str)
                if isinstance(array_data, list):
                    # 根据请求的字段名处理
                    if field_name == "Result":
                        return array_data
                    elif field_name == "Thought":
                        # 从Result数组提取第一个元素作为思考内容
                        if len(array_data) > 0:
                            thought = str(array_data[0]).strip()
                            return thought if thought.lower() != "none" else "None"
                        return ""
                    elif field_name == "Action":
                        # 从Result数组提取第一个元素作为工具名
                        if len(array_data) > 0:
                            action = str(array_data[0]).strip()
                            return action if action.lower() != "none" else "none"
                        return ""
            except (json.JSONDecodeError, ValueError):
                # 如果JSON解析失败，尝试使用ast.literal_eval（更宽松）
                try:
                    import ast
                    array_data = ast.literal_eval(array_str)
                    if isinstance(array_data, list):
                        if field_name == "Result":
                            return array_data
                        elif field_name == "Thought":
                            if len(array_data) > 0:
                                thought = str(array_data[0]).strip()
                                return thought if thought.lower() != "none" else "None"
                            return ""
                        elif field_name == "Action":
                            if len(array_data) > 0:
                                action = str(array_data[0]).strip()
                                return action if action.lower() != "none" else "none"
                            return ""
                except (ValueError, SyntaxError):
                    continue

    # 方法2: 尝试从整个文本中查找数组（更宽松的方法）
    # 查找所有可能的数组格式
    if field_name == "Result" or field_name in ["Thought", "Action"]:
        array_patterns = [
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # 匹配嵌套数组
        ]
        for pattern in array_patterns:
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            for array_str in matches:
                try:
                    array_data = json.loads(array_str)
                    if isinstance(array_data, list) and len(array_data) > 0:
                        # 检查数组内容是否合理（包含字符串）
                        if all(isinstance(item, str) for item in array_data):
                            if field_name == "Result":
                                return array_data
                            elif field_name == "Thought":
                                thought = str(array_data[0]).strip()
                                return thought if thought.lower() != "none" else "None"
                            elif field_name == "Action":
                                action = str(array_data[0]).strip()
                                return action if action.lower() != "none" else "none"
                except (json.JSONDecodeError, ValueError):
                    try:
                        import ast
                        array_data = ast.literal_eval(array_str)
                        if isinstance(array_data, list) and len(array_data) > 0:
                            if all(isinstance(item, str) for item in array_data):
                                if field_name == "Result":
                                    return array_data
                                elif field_name == "Thought":
                                    thought = str(array_data[0]).strip()
                                    return thought if thought.lower() != "none" else "None"
                                elif field_name == "Action":
                                    action = str(array_data[0]).strip()
                                    return action if action.lower() != "none" else "none"
                    except (ValueError, SyntaxError):
                        continue

    # 对于其他字段，使用传统的正则表达式提取
    # 支持 "FieldName: value" 格式
    pattern = rf'{field_name}:\s*(.*?)(?=\s*{field_name}:|\s*$)'
    matches = re.findall(pattern, text_content, re.IGNORECASE | re.DOTALL)
    if matches:
        result = matches[0].strip()
        # 如果结果看起来像JSON数组，尝试解析
        if field_name == "Result" and result.startswith('[') and result.endswith(']'):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, ValueError):
                pass
        return result

    # 最后尝试使用 extract_from_json_format（使用原始文本和清理后的文本）
    for text_to_search in [cleaned_text, text_content]:
        result = extract_from_json_format(text_to_search, "Result")
        if result:
            # 如果Result字段的结果看起来像数组，尝试解析
            if result.startswith('[') and result.endswith(']'):
                try:
                    array_data = json.loads(result)
                    if isinstance(array_data, list):
                        if field_name == "Result":
                            return array_data
                        elif field_name == "Thought":
                            if len(array_data) > 0:
                                thought = str(array_data[0]).strip()
                                return thought if thought.lower() != "none" else "None"
                            return ""
                        elif field_name == "Action":
                            if len(array_data) > 0:
                                action = str(array_data[0]).strip()
                                return action if action.lower() != "none" else "none"
                            return ""
                except (json.JSONDecodeError, ValueError):
                    pass
            # 如果结果不为空且不是数组，直接返回（用于其他字段）
            if field_name == "Result" and result.strip():
                return result

    # 如果所有方法都失败，返回默认值
    return "" if field_name != "Result" else []


def extract_json_scene_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    从文本中提取JSON格式的场景字典列表

    Args:
        text: 包含JSON的文本内容

    Returns:
        List[Dict]: 提取到的场景字典列表，如果提取失败则返回None
    """
    logger = logging.getLogger(__name__)

    if not text or not isinstance(text, str):
        logger.warning("输入文本为空或不是字符串")
        return None

    # 方法1: 尝试直接查找并解析JSON数组
    json_scene = _try_extract_json_array(text)
    if json_scene is not None:
        return json_scene

    # 方法2: 尝试查找markdown代码块中的JSON
    json_scene = _try_extract_from_code_block(text)
    if json_scene is not None:
        return json_scene

    # 方法3: 使用正则表达式查找可能的JSON数组
    json_scene = _try_extract_with_regex(text)
    if json_scene is not None:
        return json_scene

    # 方法4: 尝试修复常见的JSON格式问题
    json_scene = _try_fix_and_extract_json(text)
    if json_scene is not None:
        return json_scene

    logger.warning("无法从文本中提取有效的JSON场景数据")
    return None


def _try_extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """尝试直接解析JSON数组"""
    try:
        # 直接尝试解析整个文本
        data = json.loads(text.strip())
        if isinstance(data, list):
            return _validate_and_filter_scenes(data)
    except json.JSONDecodeError:
        pass
    return None


def _try_extract_from_code_block(text: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从markdown代码块中提取JSON"""
    # 匹配 ```json ... ``` 或 ``` ... ``` 格式
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',  # ``` ... ```
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    if isinstance(data, list):
                        return _validate_and_filter_scenes(data)
                except json.JSONDecodeError:
                    continue
    return None


def _try_extract_with_regex(text: str) -> Optional[List[Dict[str, Any]]]:
    """使用正则表达式查找可能的JSON数组"""
    # 匹配以 [ 开始，以 ] 结束的内容（可能是JSON数组）
    pattern = r'\[\s*{[\s\S]*?}\s*\]'
    matches = re.findall(pattern, text)

    for match in matches:
        # 尝试修复常见的JSON格式问题
        fixed_text = _fix_common_json_issues(match)
        try:
            data = json.loads(fixed_text)
            if isinstance(data, list):
                return _validate_and_filter_scenes(data)
        except json.JSONDecodeError:
            continue
    return None


def _try_fix_and_extract_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """尝试修复文本并提取JSON"""
    # 常见的修复策略
    fixed_text = _fix_common_json_issues(text)

    # 尝试提取JSON对象数组
    try:
        # 如果修复后的文本看起来像一个列表
        if fixed_text.strip().startswith('[') and fixed_text.strip().endswith(']'):
            data = json.loads(fixed_text)
            if isinstance(data, list):
                return _validate_and_filter_scenes(data)
    except json.JSONDecodeError:
        pass

    # 尝试查找和拼接多个JSON对象
    return _try_extract_and_combine_objects(text)


def _try_extract_and_combine_objects(text: str) -> Optional[List[Dict[str, Any]]]:
    """尝试提取多个JSON对象并组合成数组"""
    # 查找所有类似JSON对象的内容
    pattern = r'{\s*"[^"]*"\s*:\s*"[^"]*"[^{}]*}'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    scenes = []
    for match in matches:
        try:
            # 尝试解析单个对象
            obj = json.loads(match)
            if isinstance(obj, dict):
                scenes.append(obj)
        except json.JSONDecodeError:
            # 尝试修复常见的单引号问题
            fixed_match = match.replace("'", '"')
            try:
                obj = json.loads(fixed_match)
                if isinstance(obj, dict):
                    scenes.append(obj)
            except json.JSONDecodeError:
                continue

    return scenes if scenes else None


def _fix_common_json_issues(text: str) -> str:
    """修复常见的JSON格式问题"""
    if not text:
        return text

    fixed = text.strip()

    # 1. 修复单引号（将单引号转换为双引号，但要避免转义的单引号）
    fixed = re.sub(r"(?<!\\)'", '"', fixed)
    fixed = fixed.replace(r"\'", "'")  # 恢复转义的单引号

    # 2. 修复无引号的键（在某些不规范的JSON中）
    # 匹配模式: { key: "value" } -> { "key": "value" }
    pattern = r'{\s*(\w+)\s*:\s*"[^"]*"'

    def replace_key(match):
        key = match.group(1)
        return f'{{ "{key}": "{match.group(2)}"' if match.lastindex > 1 else f'{{ "{key}":'

    fixed = re.sub(pattern, replace_key, fixed)

    # 3. 修复末尾的逗号
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)

    # 4. 移除控制字符和BOM
    fixed = fixed.replace('\ufeff', '')  # UTF-8 BOM
    fixed = ''.join(char for char in fixed if ord(char) >= 32 or char in '\n\r\t')

    return fixed


def _validate_and_filter_scenes(scenes: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """验证并过滤场景列表"""
    if not isinstance(scenes, list):
        return None

    valid_scenes = []
    required_keys = ["时间段", "主体声音内容", "主体声音风格", "环境声音内容", "环境声音风格"]

    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue

        # 检查所有必需字段
        if all(key in scene for key in required_keys):
            # 确保所有字段都是字符串类型
            valid_scene = {}
            for key in required_keys:
                value = scene.get(key, "")
                # 如果是其他类型，转换为字符串
                if not isinstance(value, str):
                    value = str(value)
                valid_scene[key] = value.strip() if value else ""

            valid_scenes.append(valid_scene)

    return valid_scenes if valid_scenes else None


def _print_with_indent(prefix: str, content: str, tab_count: int = 2) -> None:
    """Update the shared one-line runtime status.

    Args:
        prefix: 状态内容前缀字符串
        content: 状态内容
        tab_count: 保留兼容参数，不再用于逐行缩进打印
    """
    _ = tab_count
    detail = f"{prefix}{content}"
    LiveStatus.update(theme=LiveStatus.infer_theme(detail), detail=detail)


# ============================================================================
# Task String Parsing and Formatting Utilities (for task_create.py)
# ============================================================================

def format_tool_usage_guides(guides: List[Dict[str, Any]]) -> str:
    """
    Format tool usage guides into readable string.

    Args:
        guides: List of tool usage guide dictionaries

    Returns:
        Formatted string representation
    """
    if not guides:
        return ""

    lines = []
    for i, guide in enumerate(guides, 1):
        tool_name = guide.get("tool_name", "unknown")
        usage_purpose = guide.get("usage_purpose", "")
        line = f"{i}. {tool_name} - {usage_purpose}"
        lines.append(line)

        dependencies = guide.get("dependencies", [])
        if dependencies:
            lines.append(f"   依赖: {', '.join(dependencies)}")

        parameters_guide = guide.get("parameters_guide", {})
        if parameters_guide:
            param_strs = [f"{k}: {v}" for k, v in parameters_guide.items()]
            lines.append(f"   参数: {'; '.join(param_strs)}")

    return "\n".join(lines)


def format_precautions(precautions: List[Dict[str, Any]]) -> str:
    """
    Format precautions into readable string.

    Args:
        precautions: List of precaution dictionaries

    Returns:
        Formatted string representation
    """
    if not precautions:
        return ""

    lines = []
    for i, precaution in enumerate(precautions, 1):
        category = precaution.get("category", "")
        rule_desc = precaution.get("rule_description", "")
        line = f"{i}. [{category}] {rule_desc}"
        lines.append(line)

        validation = precaution.get("validation_method", "")
        if validation:
            lines.append(f"   校验方法: {validation}")

        error_handling = precaution.get("error_handling", "")
        if error_handling:
            lines.append(f"   错误处理: {error_handling}")

    return "\n".join(lines)


def parse_tasks_from_json(
    json_data: Dict[str, Any],
    include_agent_info: bool = False
) -> List[str]:
    """
    Parse LLM-generated task JSON and convert to formatted strings.

    Converts Task objects with nested ToolUsageGuide and Precaution objects
    into human-readable formatted strings.

    Args:
        json_data: JSON data containing tasks (can be wrapped in 'tasks' key)
        include_agent_info: Whether to include agent_id in output

    Returns:
        List[str] - Formatted task strings, one per line with all details

    Example:
        >>> json_data = {
        ...     "tasks": [
        ...         {
        ...             "task_id": "task_001",
        ...             "description": "Generate lyrics",
        ...             "best_practices": ["Practice 1", "Practice 2"],
        ...             "tool_usage_guides": [
        ...                 {"tool_name": "pop_gt_lyric", "usage_purpose": "生成歌词"}
        ...             ],
        ...             "precautions": [
        ...                 {"category": "格式校验", "rule_description": "..."}
        ...             ]
        ...         }
        ...     ]
        ... }
        >>> tasks = parse_tasks_from_json(json_data)
        >>> print(tasks[0])  # Formatted task string
    """
    task_strings = []

    # Extract tasks array from JSON
    tasks_data = json_data
    if isinstance(json_data, dict):
        if "tasks" in json_data:
            tasks_data = json_data["tasks"]
        # Handle case where json_data itself might be tasks list
        elif "task_id" in json_data:
            tasks_data = [json_data]

    if not isinstance(tasks_data, list):
        tasks_data = [tasks_data]

    for task_item in tasks_data:
        if not isinstance(task_item, dict):
            continue

        # Build task string
        lines = []

        # Task ID
        task_id = task_item.get("task_id", "unknown")
        lines.append(f"task_id: {task_id}")
        lines.append("")  # Empty line for readability

        # Description
        description = task_item.get("description", "")
        if description:
            lines.append("描述:")
            lines.append(description)
            lines.append("")

        # Best Practices
        best_practices = task_item.get("best_practices", [])
        if best_practices:
            lines.append("最佳实践:")
            for i, practice in enumerate(best_practices, 1):
                lines.append(f"{i}. {practice}")
            lines.append("")

        # Tool Usage Guides
        tool_guides = task_item.get("tool_usage_guides", [])
        if tool_guides:
            lines.append("工具使用说明:")
            formatted_guides = format_tool_usage_guides(tool_guides)
            lines.append(formatted_guides)
            lines.append("")

        # Precautions
        precautions = task_item.get("precautions", [])
        if precautions:
            lines.append("注意事项:")
            formatted_precautions = format_precautions(precautions)
            lines.append(formatted_precautions)
            lines.append("")

        # Add divider
        lines.append("-" * 80)

        task_string = "\n".join(lines)
        task_strings.append(task_string)

    return task_strings


def extract_tasks_array_from_response(response_content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tasks array from LLM response which may contain markdown code blocks or raw JSON.

    Args:
        response_content: Raw response from LLM

    Returns:
        List of task dictionaries or None if parsing fails
    """
    content = response_content.strip()

    # Remove markdown code blocks
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        if json_end > json_start:
            content = content[json_start:json_end].strip()
    elif "```" in content:
        json_start = content.find("```") + 3
        json_end = content.find("```", json_start)
        if json_end > json_start:
            content = content[json_start:json_end].strip()

    try:
        data = json.loads(content)

        # Handle different JSON structures
        if isinstance(data, dict):
            if "tasks" in data:
                return data["tasks"]
            elif "task" in data:
                return data["task"] if isinstance(data["task"], list) else [data["task"]]
            else:
                return [data]
        elif isinstance(data, list):
            return data
        else:
            return None

    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse task JSON: {e}")
        return None

