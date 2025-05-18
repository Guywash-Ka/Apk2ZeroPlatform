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
        """Определяет тип API на основе названия модели."""
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
            llm_report_dir: str = "/Users/rodvlasov2003/PycharmProjects/vkr/java_analyzer_report_timetable/llm_report/",
            output_dir: str = "llm_output_timetable",
            max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Трансформирует все файлы, указанные в отчете LLM.

        Args:
            llm_report_dir: Директория с отчетом LLM (сгенерированным JavaDependencyAnalyzer)
            output_dir: Директория для сохранения преобразованных файлов
            max_workers: Максимальное количество параллельных потоков

        Returns:
            Статистика трансформации
        """
        # Проверяем существование директории с отчетом
        if not os.path.exists(llm_report_dir):
            logger.error(f"LLM report directory {llm_report_dir} does not exist")
            return self.stats

        # Загружаем общую информацию из отчета
        summary_path = os.path.join(llm_report_dir, "llm_summary.json")
        if not os.path.exists(summary_path):
            logger.error(f"Summary file not found at {summary_path}")
            return self.stats

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Создаем выходную директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Обрабатываем "potentially_independent" файлы
        potentially_independent_dir = os.path.join(llm_report_dir, "potentially_independent")
        if os.path.exists(potentially_independent_dir):
            self._process_category_files(
                potentially_independent_dir,
                os.path.join(output_dir, "potentially_independent"),
                summary.get("transformation_instructions", {}).get("potentially_independent", ""),
                "potentially_independent",
                max_workers
            )

        # Обрабатываем "fully_independent" файлы
        fully_independent_dir = os.path.join(llm_report_dir, "fully_independent")
        if os.path.exists(fully_independent_dir):
            self._process_category_files(
                fully_independent_dir,
                os.path.join(output_dir, "fully_independent"),
                summary.get("transformation_instructions", {}).get("fully_independent", ""),
                "fully_independent",
                max_workers
            )
        
        # Обрабатываем "android_dependent" файлы
        android_dependent_dir = os.path.join(llm_report_dir, "android_dependent")
        if os.path.exists(android_dependent_dir):
            self._process_category_files(
                android_dependent_dir,
                os.path.join(output_dir, "android_dependent"),
                summary.get("transformation_instructions", {}).get("android_dependent", ""),
                "android_dependent",
                max_workers
            )

        # Обновляем статистику
        self.stats["time_completed"] = time.time()
        self.stats["total_time_seconds"] = self.stats["time_completed"] - self.stats["time_started"]

        # Сохраняем итоговую статистику
        with open(os.path.join(output_dir, "transformation_stats.json"), 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Создаем отчет о трансформации
        self._generate_transformation_report(output_dir)

        logger.info(f"Transformation completed in {self.stats['total_time_seconds']:.2f} seconds")
        logger.info(
            f"Successfully transformed: {self.stats['successfully_transformed']}/{self.stats['total_files']} files")
        logger.info(f"Total tokens used: {self.stats['total_tokens_used']}")

        return self.stats

    def _process_category_files(self, category_dir: str, output_dir: str,
                                default_instructions: str, category: str, max_workers: int) -> None:
        """
        Обрабатывает файлы определенной категории.

        Args:
            category_dir: Директория с файлами категории
            output_dir: Директория для сохранения преобразованных файлов
            default_instructions: Инструкции по умолчанию для трансформации
            category: Название категории ('potentially_independent' или 'fully_independent')
            max_workers: Максимальное количество параллельных потоков
        """
        os.makedirs(output_dir, exist_ok=True)

        # Собираем все JSON файлы с отчетами
        json_files = [f for f in os.listdir(category_dir) if f.endswith('.json')]

        if not json_files:
            logger.warning(f"No files found in {category_dir}")
            return

        logger.info(f"Found {len(json_files)} files to transform in {category}")
        self.stats["total_files"] += len(json_files)

        # Параллельная обработка файлов
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for json_file in json_files:
                report_path = os.path.join(category_dir, json_file)
                # Находим соответствующий Java файл
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
        """
        Трансформирует один Java файл с использованием LLM.

        Args:
            report_path: Путь к JSON отчету о файле
            java_path: Путь к исходному Java файлу
            output_dir: Директория для сохранения преобразованного файла
            default_instructions: Инструкции по умолчанию
            category: Категория файла

        Returns:
            Информация о трансформации или None в случае ошибки
        """
        try:
            # Загружаем отчет
            with open(report_path, 'r') as f:
                report = json.load(f)

            # Загружаем исходный код
            with open(java_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Создаем имя выходного файла (сохраняя структуру пакета)
            class_name = report.get("class_name", "Unknown")
            package_path = report.get("package_name", "").replace(".", "/")

            # Подготавливаем выходную директорию, сохраняя структуру пакетов
            if package_path:
                package_output_dir = os.path.join(output_dir, package_path)
                os.makedirs(package_output_dir, exist_ok=True)
                output_file = os.path.join(package_output_dir, f"{class_name}.java")
            else:
                output_file = os.path.join(output_dir, f"{class_name}.java")

            # Подготавливаем запрос к LLM
            prompt = self._build_prompt(report, source_code, default_instructions, category)

            # Запрашиваем трансформацию у LLM
            start_time = time.time()
            transformed_code, tokens_info = self._call_llm_api(prompt)
            completion_time = time.time() - start_time

            if transformed_code:
                # Сохраняем преобразованный код
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transformed_code)

                # Логируем успех
                logger.info(f"Successfully transformed {class_name} - tokens: {tokens_info.get('total_tokens', 0)}")

                # Сохраняем метаданные о трансформации
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

                # Сохраняем метаданные для этого файла
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
        """
        Создает промпт для LLM на основе отчета и исходного кода.

        Args:
            report: JSON-отчет о файле
            source_code: Исходный код Java файла
            default_instructions: Инструкции по умолчанию
            category: Категория файла

        Returns:
            Промпт для LLM
        """
        class_name = report.get("class_name", "Unknown")

        # Основной шаблон промпта
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
        # Для потенциально независимых классов добавляем специфичные рекомендации
        if category == "potentially_independent" and "transformation_info" in report:
            transform_info = report["transformation_info"]

            # Добавляем информацию о методах, зависящих от Android
            if transform_info.get("android_dependent_methods"):
                prompt += "\n### Android-Dependent Methods to Transform\n"
                for method in transform_info["android_dependent_methods"]:
                    prompt += f"- `{method['name']}`: Uses Android APIs: {', '.join(method['android_apis'])}\n"

            # Добавляем информацию о полях Android-типов
            if transform_info.get("android_fields"):
                prompt += "\n### Android Fields to Abstract\n"
                for field in transform_info["android_fields"]:
                    prompt += f"- `{field['name']}` of type `{field['type']}`\n"

            # Добавляем конкретные рекомендации по трансформации
            if transform_info.get("recommendations"):
                prompt += "\n### Specific Transformation Recommendations\n"
                for rec in transform_info["recommendations"]:
                    prompt += f"- {rec}\n"

        # Добавляем информацию о импортах
        if "imports" in report:
            android_imports = report["imports"].get("android", [])
            if android_imports:
                prompt += "\n### Android Imports to Replace\n"
                for imp in android_imports:
                    prompt += f"- `{imp}`\n"

        # Финальные инструкции и требования к формату ответа
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
        """
        Вызывает API LLM для трансформации кода.

        Args:
            prompt: Промпт для LLM

        Returns:
            Tuple с преобразованным кодом и информацией о токенах
        """
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
        """
        Вызывает OpenAI API для трансформации кода.

        Args:
            prompt: Промпт для OpenAI

        Returns:
            Tuple с преобразованным кодом и информацией о токенах
        """

        try:
            import openai

            client = openai.OpenAI(base_url="https://api.eliza.yandex.net/raw/openai/v1", api_key=self.api_key)

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

            # Извлекаем ответ модели
            if hasattr(response, 'choices') and len(response.choices) > 0:
                result = response.choices[0].message.content

                # Удаляем обрамляющие markdown блоки кода, если они есть
                result = self._clean_code_response(result)

                # Получаем информацию о токенах
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
        """
        Вызывает Anthropic API для трансформации кода.

        Args:
            prompt: Промпт для Anthropic

        Returns:
            Tuple с преобразованным кодом и информацией о токенах
        """
        try:
            import anthropic

            client = anthropic.Anthropic(base_url="https://api.eliza.yandex.net/raw/anthropic", api_key=self.api_key)

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

            # Извлекаем ответ модели
            if hasattr(response, 'content'):
                # Для Anthropic API, content - это список блоков
                content_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                result = ''.join(content_blocks)

                # Удаляем обрамляющие markdown блоки кода, если они есть
                result = self._clean_code_response(result)

                # Получаем информацию о токенах (Anthropic предоставляет меньше деталей)
                # Приблизительный расчет на основе размера ответа
                approx_completion_tokens = len(result) // 4  # Грубая оценка

                token_info = {
                    "prompt_tokens": 0,  # Anthropic не предоставляет эту информацию
                    "completion_tokens": approx_completion_tokens,
                    "total_tokens": approx_completion_tokens  # Неполная информация
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
        """
        Вызывает DeepSeek API для трансформации кода.

        Args:
            prompt: Промпт для DeepSeek
            temperature: Параметр температуры для генерации

        Returns:
            Tuple с преобразованным кодом и информацией о токенах
        """
        try:
            import requests

            url = "https://api.eliza.yandex.net/together/v1/chat/completions"

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
    Your task is to remove Android dependencies while preserving functionality."""

            headers = {
                "authorization": f"OAuth {self.api_key}",
                "content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-ai/deepseek-v3",
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
                    # Получаем текст ответа
                    result_text = result["choices"][0]["message"]["content"]

                    # Удаляем обрамляющие markdown блоки кода, если они есть
                    clean_result = self._clean_code_response(result_text)

                    # Информация о токенах
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
        """
        Вызывает Yandex GPT API для трансформации кода.

        Args:
            prompt: Промпт для YandexGPT

        Returns:
            Tuple с преобразованным кодом и информацией о токенах
        """
        try:
            import requests

            model = "32b_aligned_quantized_202502"
            url = f'http://zeliboba.yandex-team.ru/balance/{model}/generative?api_key=public'

            system_message = """You are an expert Java developer specializing in converting Android-dependent code to platform-independent Java. 
    Your task is to remove Android dependencies while preserving functionality."""

            headers = {
                'Content-Type': 'application/json',
                'X-Model-Discovey-Oauth-Token': ""
            }

            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            data = json.dumps(data, ensure_ascii=False)
            data = data.encode('utf-8')

            response = requests.post(url, headers=headers, data=data)

            if response.status_code == 200:
                result = response.json()
                print(result)

                if "Responses" in result:
                    # Получаем текст ответа
                    result_text = result["Responses"][0]["Response"]

                    # Удаляем обрамляющие markdown блоки кода, если они есть
                    clean_result = self._clean_code_response(result_text)

                    # Информация о токенах
                    token_info = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": result["Responses"][0].get("NumTokens", 0)
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
        """
        Очищает ответ LLM, удаляя markdown обрамление и другие артефакты.

        Args:
            response: Сырой ответ от LLM

        Returns:
            Очищенный Java код
        """
        # Удаляем markdown блоки кода
        if response.startswith("```java"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        # Удаляем лишние переводы строк в конце и начале
        response = response.strip()

        return response

    def _generate_transformation_report(self, output_dir: str) -> None:
        """
        Генерирует итоговый отчет о трансформации.

        Args:
            output_dir: Директория для сохранения отчета
        """
        # Расчет общей статистики успешности
        success_rate = (self.stats["successfully_transformed"] / self.stats["total_files"] * 100) if self.stats["total_files"] > 0 else 0

        # Создаем отчет в формате Markdown
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
        # Сохраняем отчет
        report_path = os.path.join(output_dir, "transformation_report.md")
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Transformation report generated at {report_path}")

def main():
    """Основная функция для запуска трансформации из командной строки."""
    parser = argparse.ArgumentParser(description="LLM-based Java code transformer")
    parser.add_argument("--api-key", help="API key for LLM service", default="")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name (default: deepseek)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM (default: 0.1)")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers (default: 4)")

    args = parser.parse_args()

    transformer = LLMTransformer(api_key=args.api_key, model_name=args.model, temperature=args.temperature)
    stats = transformer.transform_directory(max_workers=args.max_workers)

    logger.info(
        f"Transformation completed. Success rate: {(stats['successfully_transformed'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0:.2f}%")

    return 0


class AndroidCodeExtractor:
    """
    Полный пайплайн для извлечения платформенно-независимого кода из Android приложений.
    Выполняет декомпиляцию APK, анализ зависимостей и трансформацию кода с помощью LLM.
"""


    def __init__(self, api_key: str, model_name: str = "gpt-4", max_workers: int = 4):
        """
        Инициализация экстрактора кода.

        Args:
            api_key: API ключ для LLM сервиса
            model_name: Название модели для использования
            max_workers: Максимальное количество параллельных потоков
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max_workers
        self.output_base_dir = "android_code_extraction"


    def process_apk(self, apk_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Обрабатывает APK файл: декомпилирует, анализирует и трансформирует код.

        Args:
            apk_path: Путь к APK файлу
            output_dir: Директория для результатов (по умолчанию создается на основе имени APK)

        Returns:
            Статистика и информация о результатах обработки
        """
        if not os.path.exists(apk_path):
            logger.error(f"APK file not found: {apk_path}")
            return {"error": "APK file not found"}

        # Создаем имя выходной директории, если не указано
        if output_dir is None:
            apk_name = os.path.basename(apk_path).replace(".apk", "")
            output_dir = os.path.join(self.output_base_dir, apk_name)

        os.makedirs(output_dir, exist_ok=True)

        # Определяем пути для промежуточных результатов
        decompiled_dir = os.path.join(output_dir, "decompiled")
        analysis_dir = os.path.join(output_dir, "analysis")
        llm_report_dir = os.path.join(analysis_dir, "llm_report")
        transformed_dir = os.path.join(output_dir, "transformed")

        # Запускаем декомпиляцию APK
        logger.info(f"Decompiling APK: {apk_path}")
        decompile_result = self._decompile_apk(apk_path, decompiled_dir)

        if not decompile_result.get("success", False):
            logger.error(f"Failed to decompile APK: {decompile_result.get('error')}")
            return {"error": f"Decompilation failed: {decompile_result.get('error')}"}

        # Анализируем декомпилированный код
        logger.info(f"Analyzing decompiled code in {decompiled_dir}")
        analysis_result = self._analyze_code(decompiled_dir, analysis_dir)

        # Трансформируем код с помощью LLM
        logger.info(f"Transforming code with {self.model_name}")
        transformation_result = self._transform_code(llm_report_dir, transformed_dir)

        # Формируем итоговый отчет
        final_report = {
            "apk_path": apk_path,
            "output_directory": output_dir,
            "decompilation": decompile_result,
            "analysis": analysis_result,
            "transformation": transformation_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Сохраняем итоговый отчет
        with open(os.path.join(output_dir, "extraction_report.json"), 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Android code extraction completed. Results saved in {output_dir}")
        return final_report

        def process_java_directory(self, java_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
            """
            Обрабатывает директорию с Java файлами: анализирует и трансформирует код.
            Этот метод пропускает этап декомпиляции, полезен для работы с существующей кодовой базой.

            Args:
                java_dir: Директория с Java файлами
                output_dir: Директория для результатов

            Returns:
                Статистика и информация о результатах обработки
            """
            if not os.path.exists(java_dir):
                logger.error(f"Java directory not found: {java_dir}")
                return {"error": "Java directory not found"}

            # Создаем имя выходной директории, если не указано
            if output_dir is None:
                dir_name = os.path.basename(java_dir)
                output_dir = os.path.join(self.output_base_dir, f"{dir_name}_extraction")

            os.makedirs(output_dir, exist_ok=True)

            # Определяем пути для промежуточных результатов
            analysis_dir = os.path.join(output_dir, "analysis")
            llm_report_dir = os.path.join(analysis_dir, "llm_report")
            transformed_dir = os.path.join(output_dir, "transformed")

            # Анализируем Java код
            logger.info(f"Analyzing Java code in {java_dir}")
            analysis_result = self._analyze_code(java_dir, analysis_dir)

            # Трансформируем код с помощью LLM
            logger.info(f"Transforming code with {self.model_name}")
            transformation_result = self._transform_code(llm_report_dir, transformed_dir)

            # Формируем итоговый отчет
            final_report = {
                "java_directory": java_dir,
                "output_directory": output_dir,
                "analysis": analysis_result,
                "transformation": transformation_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Сохраняем итоговый отчет
            with open(os.path.join(output_dir, "extraction_report.json"), 'w') as f:
                json.dump(final_report, f, indent=2)

            logger.info(f"Java code transformation completed. Results saved in {output_dir}")
            return final_report

        def _decompile_apk(self, apk_path: str, output_dir: str) -> Dict[str, Any]:
            """
            Декомпилирует APK файл, используя JADX.

            Args:
                apk_path: Путь к APK файлу
                output_dir: Директория для декомпилированного кода

            Returns:
                Результат декомпиляции
            """
            try:
                import subprocess

                logger.info(f"Decompiling {apk_path} to {output_dir}")
                start_time = time.time()

                # Проверяем наличие JADX
                try:
                    subprocess.check_output(["jadx", "--version"], stderr=subprocess.STDOUT)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.error("JADX not found. Please install JADX and add it to your PATH.")
                    return {"success": False, "error": "JADX not found. Please install JADX and add it to your PATH."}

                # Запуск JADX с оптимальными опциями
                cmd = [
                    "jadx",
                    "--deobf",  # Деобфускация
                    "--deobf-min", "3",  # Минимальная длина имени для деобфускации
                    "--deobf-use-sourcename",  # Использовать имена из исходного кода
                    "--escape-unicode",  # Экранировать юникод
                    "--threads-count", str(self.max_workers),  # Параллельная обработка
                    "-d", output_dir,  # Выходная директория
                    apk_path  # Путь к APK
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
            """
            Анализирует Java код с помощью JavaDependencyAnalyzer.

            Args:
                java_dir: Директория с Java файлами
                output_dir: Директория для результатов анализа

            Returns:
                Результат анализа
            """
            try:
                from java_analyzer import JavaDependencyAnalyzer

                logger.info(f"Analyzing Java code in {java_dir}")
                start_time = time.time()

                # Создаем и запускаем анализатор
                analyzer = JavaDependencyAnalyzer()
                analyzer.analyze_directory(java_dir, max_workers=self.max_workers)

                # Генерируем стандартный отчет
                analyzer.generate_analysis_report(output_dir)

                # Генерируем отчет для LLM
                llm_output_dir = os.path.join(output_dir, "llm_report")
                analyzer.generate_llm_transform_report(llm_output_dir)

                # Генерируем детальный отчет
                dependency_output_dir = os.path.join(output_dir, "dependency_report")
                analyzer.generate_detailed_dependency_report(dependency_output_dir)

                execution_time = time.time() - start_time

                # Формируем результат анализа
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
            """
            Трансформирует код с помощью LLM.

            Args:
                llm_report_dir: Директория с отчетом для LLM
                output_dir: Директория для преобразованного кода

            Returns:
                Результат трансформации
            """
            try:
                logger.info(f"Transforming code with {self.model_name}")
                start_time = time.time()

                # Создаем и запускаем трансформатор
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

                # Добавляем время выполнения в статистику
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

    # Главная функция для запуска всего процесса
    def extract_platform_independent_code(self):
        """Главная функция для запуска полного процесса через командную строку."""
        parser = argparse.ArgumentParser(description="Extract platform-independent code from Android applications")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Команда для обработки APK файла
        apk_parser = subparsers.add_parser("apk", help="Process APK file")
        apk_parser.add_argument("apk_path", help="Path to APK file")
        apk_parser.add_argument("--output", "-o", help="Output directory for results")

        # Команда для обработки директории с Java файлами
        java_parser = subparsers.add_parser("java", help="Process directory with Java files")
        java_parser.add_argument("java_dir", help="Directory with Java files")
        java_parser.add_argument("--output", "-o", help="Output directory for results")

        # Команда только для трансформации (пропуск анализа)
        transform_parser = subparsers.add_parser("transform", help="Transform code using LLM based on existing report")
        transform_parser.add_argument("llm_report_dir", help="Directory with LLM report")
        transform_parser.add_argument("output_dir", help="Directory for transformed code")

        # Общие аргументы
        for subparser in [apk_parser, java_parser, transform_parser]:
            subparser.add_argument("--api-key", required=True, help="API key for LLM service")
            subparser.add_argument("--model", default="gpt-4", help="LLM model name (default: gpt-4)")
            subparser.add_argument("--workers", "-w", type=int, default=4, help="Maximum number of parallel workers")

        args = parser.parse_args()

        # Проверяем, что команда указана
        if args.command is None:
            parser.print_help()
            return 1

        # Обрабатываем APK файл
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

        # Обрабатываем директорию с Java файлами
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

        # Только трансформация
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
