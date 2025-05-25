import os
import json
import time
import logging
import requests
import argparse
import traceback
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_transformer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LLMTransformer")


class LLMTransformer:
    """
    Класс для трансформации Java файлов с использованием LLM API.
    Использует отчеты, сгенерированные JavaDependencyAnalyzer, для отправки
    в LLM и получения преобразованных версий файлов.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4", temperature: float = 0.1):
        """
        Инициализация трансформатора кода.

        Args:
            api_key: API ключ для LLM сервиса (OpenAI, Anthropic и т.д.)
            model_name: Название модели для использования (по умолчанию gpt-4)
            temperature: Параметр temperature для LLM (0.1 для более детерминированных ответов)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.api_type = self._determine_api_type(model_name)
        self.stats = {
            "total_files": 0,
            "successfully_transformed": 0,
            "failed_transformations": 0,
            "time_started": time.time(),
            "time_completed": 0,
            "total_tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

    def _determine_api_type(self, model_name: str) -> str:
        if model_name.startswith(("gpt-", "text-davinci")):
            return "openai"
        elif model_name.startswith(("claude-")):
            return "anthropic"
        elif model_name.startswith(("yandexgpt")):
            return "yandex"
        elif model_name.startswith(("deepseek")):
            return "deepseek"
        elif model_name.startswith(("qwen")):
            return "qwen"
        else:
            logger.warning(f"Unknown model: {model_name}, defaulting to yandex API")
            return "yandex"

    def transform_directory(
            self,
            llm_report_dir: str = None,
            output_dir: str = None,
            max_workers: int = 4,
    ) -> Dict[str, Any]:
        load_dotenv()

        if llm_report_dir is None:
            llm_report_dir = os.getenv("LLM_REPORT_DIR", "java_analyzer_report_timetable/llm_report/")

        if output_dir is None:
            output_dir = os.getenv("OUTPUT_DIR", "llm_output_timetable")

        if not os.path.exists(llm_report_dir):
            logger.error(f"LLM report directory {llm_report_dir} does not exist")
            return self.stats

        summary_path = os.path.join(llm_report_dir, "llm_summary.json")
        if not os.path.exists(summary_path):
            logger.error(f"Summary file not found at {summary_path}")
            return self.stats

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        potentially_independent_dir = os.path.join(llm_report_dir, "potentially_independent")
        if os.path.exists(potentially_independent_dir):
            self._process_category_files(
                potentially_independent_dir,
                os.path.join(output_dir, "potentially_independent"),
                summary.get("transformation_instructions", {}).get("potentially_independent", ""),
                "potentially_independent",
                max_workers
            )

        fully_independent_dir = os.path.join(llm_report_dir, "fully_independent")
        if os.path.exists(fully_independent_dir):
            self._process_category_files(
                fully_independent_dir,
                os.path.join(output_dir, "fully_independent"),
                summary.get("transformation_instructions", {}).get("fully_independent", ""),
                "fully_independent",
                max_workers
            )

        android_dependent_dir = os.path.join(llm_report_dir, "android_dependent")
        if os.path.exists(android_dependent_dir):
            self._process_category_files(
                android_dependent_dir,
                os.path.join(output_dir, "android_dependent"),
                summary.get("transformation_instructions", {}).get("android_dependent", ""),
                "android_dependent",
                max_workers
            )

        self.stats["time_completed"] = time.time()
        self.stats["total_time_seconds"] = self.stats["time_completed"] - self.stats["time_started"]

        with open(os.path.join(output_dir, "transformation_stats.json"), 'w') as f:
            json.dump(self.stats, f, indent=2)

        self._generate_transformation_report(output_dir)

        logger.info(f"Transformation completed in {self.stats['total_time_seconds']:.2f} seconds")
        logger.info(
            f"Successfully transformed: {self.stats['successfully_transformed']}/{self.stats['total_files']} files")
        logger.info(f"Total tokens used: {self.stats['total_tokens_used']}")

        return self.stats

    def _process_category_files(self, category_dir: str, output_dir: str,
                                default_instructions: str, category: str, max_workers: int) -> None:
        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(category_dir) if f.endswith('.json')]

        if not json_files:
            logger.warning(f"No files found in {category_dir}")
            return

        logger.info(f"Found {len(json_files)} files to transform in {category}")
        self.stats["total_files"] += len(json_files)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for json_file in json_files:
                report_path = os.path.join(category_dir, json_file)
                java_file = f"{json_file.removesuffix('.json')}.java"
                java_path = os.path.join(category_dir, java_file)

                if not os.path.exists(java_path):
                    logger.warning(f"Java file not found for report {report_path} with file path {java_path}")
                    self.stats["failed_transformations"] += 1
                    continue

                future = executor.submit(
                    self._transform_single_file,
                    report_path,
                    java_path,
                    output_dir,
                    default_instructions,
                    category
                )
                futures.append(future)

            # Обрабатываем результаты
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.stats["successfully_transformed"] += 1
                        self.stats["total_tokens_used"] += int(result.get("tokens_used", 0))
                        self.stats["prompt_tokens"] += int(result.get("prompt_tokens", 0))
                        self.stats["completion_tokens"] += int(result.get("completion_tokens", 0))
                    else:
                        self.stats["failed_transformations"] += 1
                except Exception as e:
                    logger.error(f"Error in transformation task: {e}")
                    self.stats["failed_transformations"] += 1

    def _transform_single_file(self, report_path: str, java_path: str,
                               output_dir: str, default_instructions: str,
                               category: str) -> Optional[Dict[str, Any]]:
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)

            with open(java_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            class_name = report.get("class_name", "Unknown")
            package_path = report.get("package_name", "").replace(".", "/")

            if package_path:
                package_output_dir = os.path.join(output_dir, package_path)
                os.makedirs(package_output_dir, exist_ok=True)
                output_file = os.path.join(package_output_dir, f"{class_name}.java")
            else:
                output_file = os.path.join(output_dir, f"{class_name}.java")

            prompt = self._build_prompt(report, source_code, default_instructions, category)

            start_time = time.time()
            transformed_code, tokens_info = self._call_llm_api(prompt)
            completion_time = time.time() - start_time

            if transformed_code:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transformed_code)

                logger.info(f"Successfully transformed {class_name} - tokens: {tokens_info.get('total_tokens', 0)}")

                meta_data = {
                    "class_name": class_name,
                    "package_name": report.get("package_name", ""),
                    "category": category,
                    "independence_score": report.get("independence_score", 0),
                    "transformation_time_seconds": completion_time,
                    "tokens_used": tokens_info.get("total_tokens", 0),
                    "prompt_tokens": tokens_info.get("prompt_tokens", 0),
                    "completion_tokens": tokens_info.get("completion_tokens", 0),
                    "transformed_path": output_file
                }

                meta_path = output_file.replace(".java", "_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2)

                return meta_data
            else:
                logger.error(f"Failed to transform {class_name}")
                return None

        except Exception as e:
            logger.error(f"Error transforming file {java_path}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _build_prompt(self, report: Dict[str, Any], source_code: str,
                      default_instructions: str, category: str) -> str:
        class_name = report.get("class_name", "Unknown")

        prompt = f"""
# Java Code Transformation Task

## Original Code
```java
{source_code}
Transformation Instructions
You need to transform this Java code to make it platform-independent by removing Android dependencies.

Class: {class_name}
Independence Score: {report.get("independence_score", 0):.2f}
Category: {category}

{default_instructions}

"""
        if category == "potentially_independent" and "transformation_info" in report:
            transform_info = report["transformation_info"]

            if transform_info.get("android_dependent_methods"):
                prompt += "\n### Android-Dependent Methods to Transform\n"
                for method in transform_info["android_dependent_methods"]:
                    prompt += f"- `{method['name']}`: Uses Android APIs: {', '.join(method['android_apis'])}\n"

            if transform_info.get("android_fields"):
                prompt += "\n### Android Fields to Abstract\n"
                for field in transform_info["android_fields"]:
                    prompt += f"- `{field['name']}` of type `{field['type']}`\n"

            if transform_info.get("recommendations"):
                prompt += "\n### Specific Transformation Recommendations\n"
                for rec in transform_info["recommendations"]:
                    prompt += f"- {rec}\n"

        if "imports" in report:
            android_imports = report["imports"].get("android", [])
            if android_imports:
                prompt += "\n### Android Imports to Replace\n"
                for imp in android_imports:
                    prompt += f"- `{imp}`\n"

        prompt += """
Requirements for Transformed Code:
1. Remove all Android-specific imports and dependencies
2. Replace Android-specific APIs with platform-independent alternatives
3. Create abstraction interfaces for Android components when needed
4. Maintain the same package structure and class name
5. Preserve business logic and functionality
6. Ensure the code is compilable as standard Java

Additional Requirements for Kotlin-decompiled Code:
7. Remove all Kotlin-specific dependencies and imports:
   - Remove all imports starting with "kotlin."
   - Remove all imports related to "kotlinx.coroutines"
   - Remove all other Kotlin-specific packages

8. Remove all Kotlin-specific annotations:
   - Remove all @Metadata annotations
   - Remove all @DebugMetadata annotations
   - Remove all @Override annotations referring to Kotlin classes

9. Transform Kotlin coroutines to standard Java concurrency:
   - Replace CoroutineScope with java.util.concurrent.Executor or CompletableFuture
   - Replace Continuation with CompletableFuture or similar Java constructs
   - Replace SuspendLambda with appropriate Java functional interfaces
   - Replace Unit.INSTANCE with void return types or null where appropriate

10. Convert Kotlin-specific internal classes:
    - Rename classes with '$' in their names to proper Java nested classes
    - Convert anonymous inner classes to proper Java implementations

11. Remove JADX-specific artifacts:
    - Remove escaped HTML entities like "&lt;" (replace with proper characters)
    - Remove any debug or decompiler annotations

12. When dealing with inner classes:
    - Properly implement them as static nested classes when appropriate
    - Ensure proper access to outer class members

Response Format:
Return ONLY the transformed Java code as a complete file, including package declaration and imports.
Do not include explanations, markdown formatting, or code block indicators outside the Java code.
The code must be pure Java without ANY Kotlin or kotlinx dependencies.

"""
        return prompt

    def _call_llm_api(self, prompt: str) -> Tuple[Optional[str], Dict[str, int]]:
        if self.api_type == "openai":
            return self._call_openai_api(prompt)
        elif self.api_type == "anthropic":
            return self._call_anthropic_api(prompt)
        elif self.api_type == "yandex":
            return self._call_yandex_gpt_api(prompt)
        elif self.api_type == "deepseek":
            return self._call_deepseek_api(prompt)
        elif self.api_type == "qwen":
            return self._call_qwen_api(prompt)
        else:
            logger.error(f"Unsupported API type: {self.api_type}")
            return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _call_openai_api(self, prompt: str) -> Tuple[Optional[str], Dict[str, int]]:
        try:
            import openai

            client = openai.OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                   api_key=self.api_key)

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
Your task is to remove Android dependencies while preserving functionality."""
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=4000
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                result = response.choices[0].message.content

                result = self._clean_code_response(result)

                token_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

                return result, token_info
            else:
                logger.error("No valid response from OpenAI API")
                return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            logger.debug(traceback.format_exc())
            return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _call_anthropic_api(self, prompt: str) -> Tuple[Optional[str], Dict[str, int]]:
        try:
            import anthropic

            client = anthropic.Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                                         api_key=self.api_key)

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
Your task is to remove Android dependencies while preserving functionality."""
            response = client.messages.create(
                model=self.model_name,
                system=system_message,
                max_tokens=4000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            if hasattr(response, 'content'):
                content_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                result = ''.join(content_blocks)

                result = self._clean_code_response(result)

                approx_completion_tokens = len(result) // 4

                token_info = {
                    "prompt_tokens": 0,
                    "completion_tokens": approx_completion_tokens,
                    "total_tokens": approx_completion_tokens
                }

                return result, token_info
            else:
                logger.error("No valid response from Anthropic API")
                return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            logger.debug(traceback.format_exc())
            return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _call_deepseek_api(self, prompt: str, temperature: float = 0.2) -> Tuple[Optional[str], Dict[str, int]]:
        try:
            import requests

            url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
    Your task is to remove Android dependencies while preserving functionality."""

            headers = {
                "authorization": f"Bearer {self.api_key}",
                "content-Type": "application/json"
            }

            payload = {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v3"),
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()["response"]

                if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                    result_text = result["choices"][0]["message"]["content"]

                    clean_result = self._clean_code_response(result_text)

                    token_info = {
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                    }

                    return clean_result, token_info
                else:
                    logger.error(f"Unexpected response structure from DeepSeek API")
                    return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            else:
                logger.error(f"Error response from DeepSeek API: {response.status_code}, {response.text}")
                return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            logger.debug(traceback.format_exc())
            return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _call_yandex_gpt_api(self, prompt: str) -> Tuple[Optional[str], Dict[str, int]]:
        try:
            import requests

            model = os.getenv("YANDEX_MODEL", "yandexgpt")
            url = os.getenv("YANDEX_API_URL", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion")

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
    Your task is to remove Android dependencies while preserving functionality."""

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Api-Key {self.api_key}'
            }

            data = {
                "modelUri": f"gpt://{model}/latest",
                "completionOptions": {
                    "stream": False,
                    "temperature": self.temperature,
                    "maxTokens": 4000
                },
                "messages": [
                    {
                        "role": "system",
                        "text": system_message
                    },
                    {
                        "role": "user",
                        "text": prompt
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()

                if "result" in result and "alternatives" in result["result"] and len(
                        result["result"]["alternatives"]) > 0:
                    result_text = result["result"]["alternatives"][0]["message"]["text"]

                    clean_result = self._clean_code_response(result_text)

                    token_info = {
                        "prompt_tokens": result["result"].get("usage", {}).get("inputTextTokens", 0),
                        "completion_tokens": result["result"].get("usage", {}).get("completionTokens", 0),
                        "total_tokens": result["result"].get("usage", {}).get("totalTokens", 0)
                    }

                    return clean_result, token_info
                else:
                    logger.error("Unexpected response structure from YandexGPT API")
                    return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            else:
                logger.error(f"Error response from YandexGPT API: {response.status_code}, {response.text}")
                return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        except Exception as e:
            logger.error(f"Error calling YandexGPT API: {e}")
            logger.debug(traceback.format_exc())
            return None, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _clean_code_response(self, response: str) -> str:
        if response.startswith("```java"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        return response

    def _generate_transformation_report(self, output_dir: str) -> None:
        success_rate = (self.stats["successfully_transformed"] / self.stats["total_files"] * 100) if self.stats[
                                                                                                         "total_files"] > 0 else 0

        report_content = f"""# Code Transformation Report
    Summary
    Total files processed: {self.stats["total_files"]}
    Successfully transformed: {self.stats["successfully_transformed"]} ({success_rate:.2f}%)
    Failed transformations: {self.stats["failed_transformations"]}
    Total execution time: {self.stats["total_time_seconds"]:.2f} seconds
    Total tokens used: {self.stats["total_tokens_used"]}
    Prompt tokens: {self.stats["prompt_tokens"]}
    Completion tokens: {self.stats["completion_tokens"]}
    LLM Configuration
    Model: {self.model_name}
    API Type: {self.api_type}
    Temperature: {self.temperature}
    Transformed Files
    The transformed files are organized in the following directories:

    potentially_independent/: Files that required significant transformation
    fully_independent/: Files that were already mostly platform-independent
    Each transformed file has an accompanying _meta.json file with transformation metadata.

    Next Steps
    Review the transformed code to ensure it meets your requirements

    Compile and test the code to verify functionality

    Integrate the platform-independent components into your project
    """
        report_path = os.path.join(output_dir, "transformation_report.md")
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Transformation report generated at {report_path}")

def main():

    parser = argparse.ArgumentParser(description="LLM-based Java code transformer")
    parser.add_argument("--api-key", help="API key for LLM service", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o"), help="LLM model name")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                        help="Temperature for LLM")
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")),
                        help="Maximum number of parallel workers")
    parser.add_argument("--llm-report-dir", help="Directory with LLM report", default=os.getenv("LLM_REPORT_DIR"))
    parser.add_argument("--output-dir", help="Output directory for transformed files",
                        default=os.getenv("OUTPUT_DIR"))

    args = parser.parse_args()

    transformer = LLMTransformer(api_key=args.api_key, model_name=args.model, temperature=args.temperature)
    stats = transformer.transform_directory(
        llm_report_dir=args.llm_report_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )

    logger.info(
        f"Transformation completed. Success rate: {(stats['successfully_transformed'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0:.2f}%")

    return 0


class AndroidCodeExtractor:
    def __init__(self, api_key: str, model_name: str = "gpt-4", max_workers: int = 4):
        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max_workers
        self.output_base_dir = "android_code_extraction"

    def process_apk(self, apk_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(apk_path):
            logger.error(f"APK file not found: {apk_path}")
            return {"error": "APK file not found"}

        if output_dir is None:
            apk_name = os.path.basename(apk_path).replace(".apk", "")
            output_dir = os.path.join(self.output_base_dir, apk_name)

        os.makedirs(output_dir, exist_ok=True)

        decompiled_dir = os.path.join(output_dir, "decompiled")
        analysis_dir = os.path.join(output_dir, "analysis")
        llm_report_dir = os.path.join(analysis_dir, "llm_report")
        transformed_dir = os.path.join(output_dir, "transformed")

        logger.info(f"Decompiling APK: {apk_path}")
        decompile_result = self._decompile_apk(apk_path, decompiled_dir)

        if not decompile_result.get("success", False):
            logger.error(f"Failed to decompile APK: {decompile_result.get('error')}")
            return {"error": f"Decompilation failed: {decompile_result.get('error')}"}

        logger.info(f"Analyzing decompiled code in {decompiled_dir}")
        analysis_result = self._analyze_code(decompiled_dir, analysis_dir)

        logger.info(f"Transforming code with {self.model_name}")
        transformation_result = self._transform_code(llm_report_dir, transformed_dir)

        final_report = {
            "apk_path": apk_path,
            "output_directory": output_dir,
            "decompilation": decompile_result,
            "analysis": analysis_result,
            "transformation": transformation_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(output_dir, "extraction_report.json"), 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Android code extraction completed. Results saved in {output_dir}")
        return final_report

        def process_java_directory(self, java_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
            if not os.path.exists(java_dir):
                logger.error(f"Java directory not found: {java_dir}")
                return {"error": "Java directory not found"}

            if output_dir is None:
                dir_name = os.path.basename(java_dir)
                output_dir = os.path.join(self.output_base_dir, f"{dir_name}_extraction")

            os.makedirs(output_dir, exist_ok=True)

            analysis_dir = os.path.join(output_dir, "analysis")
            llm_report_dir = os.path.join(analysis_dir, "llm_report")
            transformed_dir = os.path.join(output_dir, "transformed")

            logger.info(f"Analyzing Java code in {java_dir}")
            analysis_result = self._analyze_code(java_dir, analysis_dir)

            logger.info(f"Transforming code with {self.model_name}")
            transformation_result = self._transform_code(llm_report_dir, transformed_dir)

            final_report = {
                "java_directory": java_dir,
                "output_directory": output_dir,
                "analysis": analysis_result,
                "transformation": transformation_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(os.path.join(output_dir, "extraction_report.json"), 'w') as f:
                json.dump(final_report, f, indent=2)

            logger.info(f"Java code transformation completed. Results saved in {output_dir}")
            return final_report

        def _decompile_apk(self, apk_path: str, output_dir: str) -> Dict[str, Any]:
            try:
                import subprocess

                logger.info(f"Decompiling {apk_path} to {output_dir}")
                start_time = time.time()

                try:
                    subprocess.check_output(["jadx", "--version"], stderr=subprocess.STDOUT)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.error("JADX not found. Please install JADX and add it to your PATH.")
                    return {"success": False, "error": "JADX not found. Please install JADX and add it to your PATH."}

                cmd = [
                    "jadx",
                    "--deobf",
                    "--deobf-min", "3",
                    "--deobf-use-sourcename",
                    "--escape-unicode",
                    "--threads-count", str(self.max_workers),
                    "-d", output_dir,
                    apk_path
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )

                stdout, stderr = process.communicate()

                execution_time = time.time() - start_time
                success = process.returncode == 0

                if success:
                    logger.info(f"Decompilation completed in {execution_time:.2f} seconds")
                    return {
                        "success": True,
                        "output_directory": output_dir,
                        "execution_time_seconds": execution_time
                    }
                else:
                    logger.error(f"Decompilation failed: {stderr}")
                    return {
                        "success": False,
                        "error": stderr,
                        "execution_time_seconds": execution_time
                    }

            except Exception as e:
                logger.error(f"Error during decompilation: {e}")
                logger.debug(traceback.format_exc())
                return {"success": False, "error": str(e)}

        def _analyze_code(self, java_dir: str, output_dir: str) -> Dict[str, Any]:
            try:
                from java_analyzer import JavaDependencyAnalyzer

                logger.info(f"Analyzing Java code in {java_dir}")
                start_time = time.time()

                analyzer = JavaDependencyAnalyzer()
                analyzer.analyze_directory(java_dir, max_workers=self.max_workers)

                analyzer.generate_analysis_report(output_dir)

                llm_output_dir = os.path.join(output_dir, "llm_report")
                analyzer.generate_llm_transform_report(llm_output_dir)

                dependency_output_dir = os.path.join(output_dir, "dependency_report")
                analyzer.generate_detailed_dependency_report(dependency_output_dir)

                execution_time = time.time() - start_time

                stats = analyzer.stats
                result = {
                    "success": True,
                    "output_directory": output_dir,
                    "execution_time_seconds": execution_time,
                    "total_files": stats.total_files,
                    "successfully_parsed": stats.successfully_parsed,
                    "parse_errors": stats.parse_errors,
                    "android_dependent": stats.android_dependent,
                    "potentially_independent": stats.potentially_independent,
                    "fully_independent": stats.fully_independent
                }

                logger.info(f"Analysis completed in {execution_time:.2f} seconds")
                logger.info(
                    f"Found {stats.potentially_independent + stats.fully_independent + stats.android_dependent} potentially extractable classes")

                return result

            except Exception as e:
                logger.error(f"Error during code analysis: {e}")
                logger.debug(traceback.format_exc())
                return {"success": False, "error": str(e)}

        def _transform_code(self, llm_report_dir: str, output_dir: str) -> Dict[str, Any]:
            try:
                logger.info(f"Transforming code with {self.model_name}")
                start_time = time.time()

                transformer = LLMTransformer(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.1
                )

                stats = transformer.transform_directory(
                    llm_report_dir=llm_report_dir,
                    output_dir=output_dir,
                    max_workers=self.max_workers
                )

                execution_time = time.time() - start_time

                stats["execution_time_seconds"] = execution_time
                stats["success"] = True

                logger.info(f"Transformation completed in {execution_time:.2f} seconds")
                logger.info(
                    f"Successfully transformed {stats['successfully_transformed']}/{stats['total_files']} files")

                return stats

            except Exception as e:
                logger.error(f"Error during code transformation: {e}")
                logger.debug(traceback.format_exc())
                return {"success": False, "error": str(e)}

    def extract_platform_independent_code(self):
        parser = argparse.ArgumentParser(description="Extract platform-independent code from Android applications")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        apk_parser = subparsers.add_parser("apk", help="Process APK file")
        apk_parser.add_argument("apk_path", help="Path to APK file")
        apk_parser.add_argument("--output", "-o", help="Output directory for results")

        java_parser = subparsers.add_parser("java", help="Process directory with Java files")
        java_parser.add_argument("java_dir", help="Directory with Java files")
        java_parser.add_argument("--output", "-o", help="Output directory for results")

        transform_parser = subparsers.add_parser("transform", help="Transform code using LLM based on existing report")
        transform_parser.add_argument("llm_report_dir", help="Directory with LLM report")
        transform_parser.add_argument("output_dir", help="Directory for transformed code")

        for subparser in [apk_parser, java_parser, transform_parser]:
            subparser.add_argument("--api-key", required=True, help="API key for LLM service")
            subparser.add_argument("--model", default="gpt-4", help="LLM model name (default: gpt-4)")
            subparser.add_argument("--workers", "-w", type=int, default=4, help="Maximum number of parallel workers")

        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            return 1

        if args.command == "apk":
            extractor = AndroidCodeExtractor(
                api_key=args.api_key,
                model_name=args.model,
                max_workers=args.workers
            )
            result = extractor.process_apk(args.apk_path, args.output)

            if "error" in result:
                logger.error(f"Failed to process APK: {result['error']}")
                return 1

            logger.info(f"APK processing completed successfully")
            return 0

        elif args.command == "java":
            extractor = AndroidCodeExtractor(
                api_key=args.api_key,
                model_name=args.model,
                max_workers=args.workers
            )
            result = extractor.process_java_directory(args.java_dir, args.output)

            if "error" in result:
                logger.error(f"Failed to process Java directory: {result['error']}")
                return 1

            logger.info(f"Java directory processing completed successfully")
            return 0

        elif args.command == "transform":
            transformer = LLMTransformer(
                api_key=args.api_key,
                model_name=args.model,
                temperature=0.1
            )

            stats = transformer.transform_directory(
                llm_report_dir=args.llm_report_dir,
                output_dir=args.output_dir,
                max_workers=args.workers
            )

            if stats["successfully_transformed"] > 0:
                logger.info(f"Transformation completed successfully")
                return 0
            else:
                logger.error("No files were successfully transformed")
                return 1

        return 0

if __name__ == "__main__":
    main()
