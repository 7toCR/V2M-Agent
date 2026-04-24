import os
import json
import logging

logger = logging.getLogger(__name__)

from .JSONSchema import JSONSchema
from .system_prompt_header import *
from .system_prompt_body import *
from langchain_core.messages import AIMessage
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

_example_call: object = {
    "name": "PatrioticSceneMusic-helper(格式为:xxxx-helper)",
    "description": "我是PatrioticSceneMusic-helper，一位专注于为爱国主义朗诵、视频或舞台表演匹配与生成背景音乐的智能助手。我深入理解中国近现代革命历史与新时代成就的叙事逻辑，能精准识别文本中的情感起伏、历史意象与时代精神。我熟悉多种非传统中国风的配乐风格（如史诗管弦、现代交响、电影原声、氛围电子等），擅长根据声音内容、情绪变化和环境氛围推荐或构建契合度高、感染力强的背景音乐方案。",
    "goal":
        "基于用户提供的分段朗诵内容及其声音风格、情绪演进和环境特征，排除Chinese Tradition风格，选择一种更具现代感或国际化的合适音乐风格"
        "确保所选风格能强化从历史回溯到新时代崛起再到青春誓言的情感弧线"
        "输出明确的音乐风格名称及选择理由"
        "支持后续用音视频制作或AI配乐生成"
        "保持风格统一且避免突兀切换",

    "directives": {
        "best_practices": [
            "始终首先调用preprocess_scene工具验证JSON场景数据的格式和完整性，确保后续分析基于有效数据。",
            "如果json_scene为None或处理失败，不要直接调用pop_audio_type，应重新检查数据提取逻辑。",
            "在处理场景音乐风格选择任务时，优先考虑具有宏大叙事能力的非中国传统音乐风格，如史诗管弦或现代交响。",
            "确保音乐风格能随朗诵情绪自然演进：起始庄重深沉，中段激昂奋进，后段明亮自豪，高潮处极具爆发力。",
            "结合环境音描述和空间感要求选择具备动态范围和层次感的音乐风格。"
        ],
        "constraints": [
            "不得推荐或使用Chinese Tradition风格，即使其与红色主题常见搭配。",
            "避免使用节奏过于轻快、风格戏谑或情感基调不符的音乐类型。",
            "所选风格必须能兼容50秒内的情绪剧烈变化，不能因风格限制导致情感断层。",
            "不可引入未在主流影视或爱国宣传中验证过的实验性或小众音乐风格，确保实用性与接受度。"
        ],
    }
}


system_prompt: str = f'''
"你的任务定制面向任务的智能体以帮助帮助用户完成用户在三引号中定义的任务。"
"1.你需要为该智能体提供一个基于任务的智能体的名称."
"2.一段关于智能体功能的描述性说明."
"3.一段关于智能体功能的期望目标的描述性说明."
"4.并在'最佳实践'和'约束条件'这两个类别下分别提供 1 到 5 条指导原则."
"5.这些原则应最有效地确保其能够成功完成所指派的任务.\n"
"\n"
"以下内容仅为案例："
"任务输入:\n"
'"""请帮助我进行流行音乐的创作"""\n\n'
"结果输出:\n"
"```\n"
f"{json.dumps(_example_call, indent=4)}"
"\n```"
'''

_response_sample: object = {
    "name": JSONSchema(
        type=JSONSchema.Type.STRING,
        description="自主代理的简短角色名称。(格式严格为:功能-helper 例子:POPMusic-helper)",
        required=True,
    ),
    "description": JSONSchema(
        type=JSONSchema.Type.STRING,
        description="一到五项关于智能体功能的说明性语句。",
        required=True,
    ),
    "goal": JSONSchema(
        type=JSONSchema.Type.STRING,
        description="一到五项关于智能体的期望目标的说明性语句。",
        required=True,
    ),
    "directives": JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "best_practices": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description="一到五项高效的最佳实践，与给定任务的完成最为契合。",
                required=True,
            ),
            "constraints": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description="一到五项合理且有效的约束条件，与给定任务的完成最为契合。",
                required=True,
            ),
        },
    ),
}

# 转换JSONSchema对象为字典以便序列化
def schema_to_dict(obj):
    if isinstance(obj, JSONSchema):
        return obj.to_dict()
    return obj

response_sample = f"\n输出格式如下:{json.dumps(_response_sample, indent=4, default=schema_to_dict)}"

class SystemPrompt():
    def __init__(
        self,
        system_prompt: str=system_prompt,
        response_sample: str=response_sample,
    ):
        self._system_prompt= system_prompt
        self._response_sample =response_sample

    def build_prompt(self,task) -> str:
        system_message = self._system_prompt
        user_message = f'"""用户任务:{task}"""'
        response_message = self._response_sample
        prompt=system_message+user_message+response_message
        #print(f"prompt:\n{prompt}\n")
        return prompt

    def parse_response_content(
        self,
        response: AIMessage,
    ) -> tuple[SystemPromptHeader, SystemPromptBody]:
        """Parse the actual text response from the objective model.

        Args:
            response: The raw response from the objective model.

        Returns:
            The parsed response.
        """
        import re
        
        def fix_json_string(s: str) -> str:
            """修复常见的JSON格式问题"""
            # 1. 提取JSON对象部分
            start_idx = s.find('{')
            end_idx = s.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                s = s[start_idx:end_idx + 1]
            
            # 2. 移除控制字符（但保留必要的空白字符）
            s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
            
            # 3. 修复单引号为双引号
            # 使用正则表达式替换属性名和属性值的单引号，但要避免替换字符串内的引号
            def replace_single_quotes(text):
                """将属性名和属性值的单引号替换为双引号"""
                result = []
                i = 0
                in_string = False
                escape_next = False
                
                while i < len(text):
                    char = text[i]
                    
                    if escape_next:
                        result.append(char)
                        escape_next = False
                    elif char == '\\':
                        result.append(char)
                        escape_next = True
                    elif char == '"':
                        in_string = not in_string
                        result.append(char)
                    elif char == "'":
                        if in_string:
                            # 在双引号字符串内，保持单引号原样
                            result.append(char)
                        else:
                            # 在字符串外，替换为双引号
                            result.append('"')
                    else:
                        result.append(char)
                    i += 1
                
                return ''.join(result)
            
            s = replace_single_quotes(s)
            
            # 4. 修复未转义的换行符在字符串值中
            # 使用更安全的方法：在字符串值中，将未转义的换行符转义
            lines = []
            in_string = False
            escape_next = False
            i = 0
            
            while i < len(s):
                char = s[i]
                
                if escape_next:
                    lines.append(char)
                    escape_next = False
                elif char == '\\':
                    lines.append(char)
                    escape_next = True
                elif char == '"' and (i == 0 or s[i-1] != '\\'):
                    in_string = not in_string
                    lines.append(char)
                elif in_string and char == '\n':
                    lines.append('\\n')
                elif in_string and char == '\r':
                    lines.append('\\r')
                elif in_string and char == '\t':
                    lines.append('\\t')
                else:
                    lines.append(char)
                i += 1
            
            s = ''.join(lines)
            
            # 5. 移除尾随逗号
            s = re.sub(r',(\s*[}\]])', r'\1', s)
            
            return s
        
        try:
            # 提取JSON内容，从```json代码块中解析
            content = response.content.strip()
            logger.debug(f"Raw response content: {content[:500]}...")  # Debug output
            if not content:
                raise ValueError("Response content is empty")

            # 查找并提取JSON代码块
            if "```json" in content:
                # 找到```json的开始位置
                json_start = content.find("```json")
                if json_start != -1:
                    # 移除```json之前的部分
                    content = content[json_start + 7:]
                    # 找到下一个```的位置
                    json_end = content.find("```")
                    if json_end != -1:
                        content = content[:json_end]
            elif "```" in content:
                # 处理以```开头的代码块
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            content = content.strip()

            # 尝试直接解析
            arguments = None
            last_error = None
            
            # 尝试1: 直接解析
            try:
                arguments = json.loads(content)
            except json.JSONDecodeError as e:
                last_error = e
                logger.debug(f"Direct JSON parsing failed: {e}")
                
                # 尝试2: 使用修复函数
                try:
                    fixed_content = fix_json_string(content)
                    logger.debug(f"Fixed content (first 500 chars): {repr(fixed_content[:500])}")
                    arguments = json.loads(fixed_content)
                except json.JSONDecodeError as e2:
                    last_error = e2
                    logger.debug(f"Fixed JSON parsing failed: {e2}")
                    
                    # 尝试3: 移除所有空白字符后解析
                    try:
                        compact = re.sub(r'\s+', ' ', content)
                        compact = fix_json_string(compact)
                        arguments = json.loads(compact)
                    except json.JSONDecodeError as e3:
                        last_error = e3
                        logger.debug(f"Compact JSON parsing failed: {e3}")
                        
                        # 尝试4: 使用ast.literal_eval作为最后手段（仅Python对象）
                        try:
                            import ast
                            # 将单引号替换为双引号（简单版本）
                            python_like = content.replace("'", '"')
                            # 尝试解析
                            parsed = ast.literal_eval(python_like)
                            # 转换为JSON兼容格式
                            arguments = json.loads(json.dumps(parsed))
                        except Exception as e4:
                            # 打印详细错误信息以便调试
                            print(f"\n{'='*80}")
                            print("JSON解析失败 - 所有尝试都失败了")
                            print(f"{'='*80}")
                            print(f"最后错误: {last_error}")
                            print(f"\n原始响应内容 (前1000字符):")
                            print(f"{repr(content[:1000])}")
                            print(f"\n完整响应内容:")
                            print(f"{content}")
                            print(f"{'='*80}\n")
                            logger.error(f"All JSON parsing attempts failed. Last error: {last_error}")
                            logger.error(f"Content (first 1000 chars): {repr(content[:1000])}")
                            raise ValueError(f"Could not parse JSON response. Last error: {last_error}. Content preview: {repr(content[:200])}...") from last_error

            # Parse goal string into list by splitting on semicolons or periods
            goal_string = arguments.get("goal", "")
            if isinstance(goal_string, str):
                # Split by semicolons first, then by periods if no semicolons
                if ";" in goal_string:
                    agent_goals = [goal.strip() for goal in goal_string.split(";") if goal.strip()]
                else:
                    # Split by periods and clean up
                    goals = [goal.strip() for goal in goal_string.split("。") if goal.strip()]
                    agent_goals = [goal + "。" for goal in goals if not goal.endswith("。")]
            else:
                agent_goals = []

            system_prompt_header = SystemPromptHeader(
                agent_name=arguments.get("name"),  # type: ignore
                agent_role=arguments.get("description"),  # type: ignore
                agent_goals=agent_goals,  # type: ignore
            )
            system_prompt_body = SystemPromptBody(
                best_practices=arguments.get("directives", {}).get("best_practices", []),
                constraints=arguments.get("directives", {}).get("constraints", []),
                resources=[],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse this response content: {response.content}")
            logger.debug(f"Error: {e}")
            raise ValueError(f"agent profile creation failed: {e}")
        return system_prompt_header, system_prompt_body

model = ChatOpenAI(
    model="qwen3-max",
    api_key=os.getenv("MCP_API_KEY"),
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=1,
)

async def generate_other_system_prompt_for_task(
    task: str,
    model:ChatOpenAI=model,
) -> tuple[SystemPromptHeader, SystemPromptBody]:
    """Generates an AIConfig object from the given string.

    Returns:
    tuple[SystemPromptHeader, SystemPromptBody]: The parsed response containing header and body
    """
    systemprompt = SystemPrompt()

    prompt = systemprompt.build_prompt(task)

    response = await model.ainvoke(prompt)
    print(f"response: {response}\n")

    return systemprompt.parse_response_content(response)


# 测试函数
async def main():
    """测试生成代理配置文件的功能"""
    task1="帮我创作古典音乐"
    task2="检索学术资料"
    try:
        task = task2
        print(f"输入任务: {task}")
        print("正在生成代理配置文件...\n")

        header, body = await generate_other_system_prompt_for_task(task)

        print("=" * 60)
        print("生成的代理配置文件:")
        print("=" * 60)
        print(f"代理名称: {header.agent_name}")
        print(f"\n代理角色描述:")
        print(f"{header.agent_role}")
        print(f"\n代理目标 ({len(header.agent_goals)} 项):")
        for i, goal in enumerate(header.agent_goals, 1):
            print(f"  {i}. {goal}")
        print(f"\n最佳实践 ({len(body.best_practices)} 项):")
        for i, practice in enumerate(body.best_practices, 1):
            print(f"  {i}. {practice}")
        print(f"\n约束条件 ({len(body.constraints)} 项):")
        for i, constraint in enumerate(body.constraints, 1):
            print(f"  {i}. {constraint}")
        print(f"\n可用资源: {body.resources if body.resources else '无'}")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())