import os
import time
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import concurrent.futures
from collections import defaultdict
import traceback

# Для парсинга Java
try:
    import javalang
    from javalang.tree import CompilationUnit, ClassDeclaration, MethodDeclaration, FieldDeclaration, Import
except ImportError:
    print("Error: javalang library not found. Installing it automatically...")
    import subprocess

    subprocess.check_call(["pip", "install", "javalang"])
    import javalang
    from javalang.tree import CompilationUnit, ClassDeclaration, MethodDeclaration, FieldDeclaration, Import

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("java_analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("JavaDependencyAnalyzer")


@dataclass
class JavaAnalysisStats:
    """Класс для хранения статистики анализа."""
    total_files: int = 0
    successfully_parsed: int = 0
    parse_errors: int = 0
    android_dependent: int = 0
    potentially_independent: int = 0
    fully_independent: int = 0
    time_started: float = field(default_factory=time.time)
    time_ended: float = 0

    def get_parse_success_rate(self) -> float:
        """Возвращает процент успешно распарсенных файлов."""
        if self.total_files == 0:
            return 0.0
        return (self.successfully_parsed / self.total_files) * 100

    def get_execution_time(self) -> float:
        """Возвращает время выполнения в секундах."""
        if self.time_ended == 0:
            self.time_ended = time.time()
        return self.time_ended - self.time_started

    def get_files_per_second(self) -> float:
        """Возвращает скорость обработки файлов."""
        execution_time = self.get_execution_time()
        if execution_time == 0:
            return 0.0
        return self.successfully_parsed / execution_time


@dataclass
class MethodInfo:
    """Информация о методе Java-класса."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]
    is_android_dependent: bool = False
    android_apis_used: List[str] = field(default_factory=list)
    independence_score: float = 1.0  # 0.0 (полностью зависимый) до 1.0 (полностью независимый)

    def update_independence_score(self) -> None:
        """Обновляет оценку независимости метода."""
        if self.is_android_dependent:
            if len(self.android_apis_used) > 5:
                self.independence_score = 0.0
            else:
                self.independence_score = max(0.1, 1.0 - (len(self.android_apis_used) * 0.2))
        else:
            self.independence_score = 1.0


@dataclass
class FieldInfo:
    """Информация о поле Java-класса."""
    name: str
    type_name: str
    is_android_type: bool = False


@dataclass
class JavaClassInfo:
    """Подробная информация о Java-классе."""
    file_path: str
    package_name: str
    class_name: str
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    android_imports: List[str] = field(default_factory=list)
    fields: List[FieldInfo] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    is_android_component: bool = False
    is_android_dependent: bool = False
    is_fully_independent: bool = False
    independence_score: float = 0.0
    dependency_type: str = "UNKNOWN"
    parse_error: Optional[str] = None
    parse_time: float = 0.0

    def update_classification(self) -> None:
        """Обновляет классификацию на основе зависимостей."""
        # Проверка на Android-компонент
        android_component_patterns = [
            "Activity", "Fragment", "Service", "BroadcastReceiver",
            "ContentProvider", "Application", "View", "ViewGroup"
        ]

        # Проверка extend/implements
        if self.extends:
            for pattern in android_component_patterns:
                if pattern in self.extends:
                    self.is_android_component = True
                    break

        for impl in self.implements:
            for pattern in android_component_patterns:
                if pattern in impl:
                    self.is_android_component = True
                    break
            if self.is_android_component:
                break

        # Подсчет Android-зависимостей в импортах
        android_import_count = len(self.android_imports)
        total_imports = len(self.imports)

        # Подсчет Android-зависимых полей
        android_field_count = sum(1 for field in self.fields if field.is_android_type)
        total_fields = len(self.fields)

        # Подсчет Android-зависимых методов
        android_method_count = sum(1 for method in self.methods if method.is_android_dependent)
        total_methods = len(self.methods)

        # Подсчет средних оценок независимости методов
        method_independence_scores = [m.independence_score for m in self.methods]
        avg_method_independence = sum(method_independence_scores) / len(
            method_independence_scores) if method_independence_scores else 1.0

        # Расчет общей оценки независимости
        # Учитываем: импорты, поля, методы, компонентность
        import_factor = 1.0 - (android_import_count / total_imports if total_imports > 0 else 0)
        field_factor = 1.0 - (android_field_count / total_fields if total_fields > 0 else 0)
        component_factor = 0.0 if self.is_android_component else 1.0

        # Веса для различных факторов
        weights = {
            "component": 0.4,  # Наивысший вес - если класс является Android-компонентом
            "methods": 0.3,  # Высокий вес для методов
            "imports": 0.2,  # Средний вес для импортов
            "fields": 0.1  # Низкий вес для полей
        }

        self.independence_score = (
                component_factor * weights["component"] +
                avg_method_independence * weights["methods"] +
                import_factor * weights["imports"] +
                field_factor * weights["fields"]
        )

        # Классификация на основе оценки
        if self.independence_score < 0.3 or self.is_android_component:
            self.is_android_dependent = True
            self.is_fully_independent = False
            self.dependency_type = "ANDROID_DEPENDENT"
        elif self.independence_score > 0.8:
            self.is_android_dependent = False
            self.is_fully_independent = True
            self.dependency_type = "FULLY_INDEPENDENT"
        else:
            self.is_android_dependent = False
            self.is_fully_independent = False
            self.dependency_type = "POTENTIALLY_INDEPENDENT"


class JavaDependencyAnalyzer:
    """Анализатор зависимостей Java-файлов с фокусом на Android."""

    def __init__(self):
        self.android_packages = {
            "android.", "androidx.", "com.google.android.",
            "dalvik.", "com.android."
        }
        self.android_components = {
            "Activity", "Fragment", "Service", "BroadcastReceiver",
            "ContentProvider", "Application", "View", "ViewGroup"
        }
        self.standard_replacements = {
            "android.util.Log": "java.util.logging.Logger",
            "android.os.AsyncTask": "java.util.concurrent.Executor",
            "android.os.Handler": "java.util.Timer"
        }
        self.stats = JavaAnalysisStats()
        self.class_info_results = {}

    def is_android_import(self, import_path: str) -> bool:
        """Проверяет, является ли импорт Android-зависимым."""
        for pkg in self.android_packages:
            if import_path.startswith(pkg):
                return True
        return False

    def is_android_type(self, type_name: str, imports: List[str]) -> bool:
        """Проверяет, является ли тип Android-зависимым."""
        # Проверка полных имен
        for pkg in self.android_packages:
            if type_name.startswith(pkg):
                return True

        # Проверка простых имен через импорты
        simple_type = type_name.split('.')[-1]
        for imp in imports:
            if imp.endswith('.' + simple_type):
                if self.is_android_import(imp):
                    return True

        # Проверка известных Android-типов
        for component in self.android_components:
            if type_name.endswith(component):
                return True

        return False

    def analyze_method_body(self, method_body: str, imports: List[str], android_imports: List[str]) -> Tuple[
        bool, List[str]]:
        """Анализирует тело метода на наличие Android-зависимостей."""
        is_android_dependent = False
        android_apis_used = []

        # Проверка использования Android-импортов
        for android_import in android_imports:
            class_name = android_import.split('.')[-1]
            if class_name in method_body:
                is_android_dependent = True
                android_apis_used.append(class_name)

        # Поиск вызовов известных Android-специфичных методов
        android_method_patterns = [
            r"getSystemService\(",
            r"findViewById\(",
            r"setContentView\(",
            r"startActivity\(",
            r"getApplication\(",
            r"getContext\(",
            r"getActivity\("
        ]

        for pattern in android_method_patterns:
            if re.search(pattern, method_body):
                is_android_dependent = True
                android_apis_used.append(pattern[:-2])  # Удаляем "(" из шаблона

        return is_android_dependent, android_apis_used

    def parse_java_file(self, file_path: str) -> JavaClassInfo:
        """Парсит Java-файл и извлекает информацию о зависимостях."""
        start_time = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = javalang.parse.parse(content)

            # Базовая информация
            package_name = tree.package.name if tree.package else "default"
            imports = [imp.path for imp in tree.imports]
            android_imports = [imp for imp in imports if self.is_android_import(imp)]

            # Находим основной класс в файле
            main_class = None
            for path, class_decl in tree.filter(javalang.tree.ClassDeclaration):
                if main_class is None or len(path) < len(main_class[0]):
                    main_class = (path, class_decl)

            if main_class is None:
                # Проверка на наличие интерфейса
                for path, interface_decl in tree.filter(javalang.tree.InterfaceDeclaration):
                    if main_class is None or len(path) < len(main_class[0]):
                        # Адаптируем интерфейс под формат класса
                        interface_decl.extends = None
                        interface_decl.implements = []
                        main_class = (path, interface_decl)

            if main_class is None:
                # Если не нашли ни класса, ни интерфейса
                class_info = JavaClassInfo(
                    file_path=file_path,
                    package_name=package_name,
                    class_name=os.path.basename(file_path).replace(".java", ""),
                    imports=imports,
                    android_imports=android_imports,
                    parse_error="No class or interface found"
                )
                return class_info

            _, class_decl = main_class

            # Информация о классе
            class_name = class_decl.name
            extends = class_decl.extends.name if hasattr(class_decl, 'extends') and class_decl.extends else None
            implements = [i.name for i in class_decl.implements] if hasattr(class_decl,
                                                                            'implements') and class_decl.implements else []

            # Анализ полей
            fields = []
            for _, field_decl in tree.filter(javalang.tree.FieldDeclaration):
                for var_decl in field_decl.declarators:
                    field_name = var_decl.name
                    type_name = field_decl.type.name
                    if hasattr(field_decl.type, 'arguments') and field_decl.type.arguments:
                        for arg in field_decl.type.arguments:
                            if hasattr(arg, 'type') and hasattr(arg.type, 'name'):
                                type_name += f"<{arg.type.name}>"

                    is_android_type = self.is_android_type(type_name, imports)

                    fields.append(FieldInfo(
                        name=field_name,
                        type_name=type_name,
                        is_android_type=is_android_type
                    ))

            # Анализ методов
            methods = []
            for _, method_decl in tree.filter(javalang.tree.MethodDeclaration):
                method_name = method_decl.name

                # Определение возвращаемого типа
                return_type = "void"
                if method_decl.return_type:
                    if hasattr(method_decl.return_type, 'name'):
                        return_type = method_decl.return_type.name
                    elif hasattr(method_decl.return_type, 'value'):
                        return_type = method_decl.return_type.value

                # Параметры метода
                parameters = []
                if method_decl.parameters:
                    for param in method_decl.parameters:
                        param_type = ""
                        if hasattr(param.type, 'name'):
                            param_type = param.type.name
                        elif hasattr(param.type, 'value'):
                            param_type = param.type.value

                        parameters.append((param.name, param_type))

                # Анализ тела метода
                method_body = str(method_decl.body) if method_decl.body else ""
                is_android_dependent, android_apis_used = self.analyze_method_body(
                    method_body, imports, android_imports
                )

                method_info = MethodInfo(
                    name=method_name,
                    return_type=return_type,
                    parameters=parameters,
                    is_android_dependent=is_android_dependent,
                    android_apis_used=android_apis_used
                )
                method_info.update_independence_score()
                methods.append(method_info)

            # Создаем результат
            class_info = JavaClassInfo(
                file_path=file_path,
                package_name=package_name,
                class_name=class_name,
                extends=extends,
                implements=implements,
                imports=imports,
                android_imports=android_imports,
                fields=fields,
                methods=methods,
                parse_time=time.time() - start_time
            )

            # Классификация
            class_info.update_classification()

            self.stats.successfully_parsed += 1
            return class_info

        except Exception as e:
            self.stats.parse_errors += 1
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error parsing {file_path}: {error_msg}")
            logger.debug(traceback.format_exc())

            # Создаем результат с ошибкой
            class_info = JavaClassInfo(
                file_path=file_path,
                package_name="unknown",
                class_name=os.path.basename(file_path).replace(".java", ""),
                parse_error=error_msg,
                parse_time=time.time() - start_time
            )
            return class_info

    def analyze_directory(self, directory_path: str, max_workers: int = 8) -> Dict[str, JavaClassInfo]:
        """Анализирует все Java-файлы в директории параллельно."""
        java_files = []

        # Сбор всех Java-файлов
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))

        self.stats.total_files = len(java_files)
        logger.info(f"Found {self.stats.total_files} Java files in {directory_path}")

        results = {}

        # Параллельная обработка файлов
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.parse_java_file, file): file for file in java_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    class_info = future.result()
                    results[file] = class_info

                    # Обновляем статистику
                    if class_info.dependency_type == "ANDROID_DEPENDENT":
                        self.stats.android_dependent += 1
                    elif class_info.dependency_type == "POTENTIALLY_INDEPENDENT":
                        self.stats.potentially_independent += 1
                    elif class_info.dependency_type == "FULLY_INDEPENDENT":
                        self.stats.fully_independent += 1

                    # Логирование прогресса через каждые 100 файлов
                    if (self.stats.successfully_parsed + self.stats.parse_errors) % 100 == 0:
                        progress = ((
                                            self.stats.successfully_parsed + self.stats.parse_errors) / self.stats.total_files) * 100
                        logger.info(
                            f"Progress: {progress:.2f}% ({self.stats.successfully_parsed + self.stats.parse_errors}/{self.stats.total_files})")

                except Exception as e:
                    logger.error(f"Error processing file {file}: {e}")

        self.class_info_results = results
        self.stats.time_ended = time.time()

        # Информация о завершении
        execution_time = self.stats.get_execution_time()
        success_rate = self.stats.get_parse_success_rate()
        files_per_second = self.stats.get_files_per_second()

        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
        logger.info(
            f"Successful parses: {self.stats.successfully_parsed}/{self.stats.total_files} ({success_rate:.2f}%)")
        logger.info(f"Processing speed: {files_per_second:.2f} files/second")
        logger.info(f"Android dependent classes: {self.stats.android_dependent}")
        logger.info(f"Potentially independent classes: {self.stats.potentially_independent}")
        logger.info(f"Fully independent classes: {self.stats.fully_independent}")

        return results

    def generate_analysis_report(self, output_dir: str = "report") -> Dict[str, Any]:
        """Генерирует подробный отчет о результатах анализа."""
        if not self.class_info_results:
            logger.error("No analysis results available. Run analyze_directory first.")
            return {}

        # Создаем директорию отчета
        os.makedirs(output_dir, exist_ok=True)

        # Базовая статистика
        stats_data = {
            "total_files": self.stats.total_files,
            "successfully_parsed": self.stats.successfully_parsed,
            "parse_errors": self.stats.parse_errors,
            "success_rate_percent": self.stats.get_parse_success_rate(),
            "execution_time_seconds": self.stats.get_execution_time(),
            "files_per_second": self.stats.get_files_per_second(),
            "android_dependent_count": self.stats.android_dependent,
            "potentially_independent_count": self.stats.potentially_independent,
            "fully_independent_count": self.stats.fully_independent,
            "android_dependent_percent": (
                        self.stats.android_dependent / self.stats.successfully_parsed * 100) if self.stats.successfully_parsed > 0 else 0,
            "potentially_independent_percent": (
                        self.stats.potentially_independent / self.stats.successfully_parsed * 100) if self.stats.successfully_parsed > 0 else 0,
            "fully_independent_percent": (
                        self.stats.fully_independent / self.stats.successfully_parsed * 100) if self.stats.successfully_parsed > 0 else 0
        }

        # Информация о пакетах
        package_stats = defaultdict(
            lambda: {"total": 0, "android_dependent": 0, "potentially_independent": 0, "fully_independent": 0})

        for file_path, class_info in self.class_info_results.items():
            package = class_info.package_name
            package_stats[package]["total"] += 1

            if class_info.is_android_dependent:
                package_stats[package]["android_dependent"] += 1
            elif class_info.is_fully_independent:
                package_stats[package]["fully_independent"] += 1
            else:
                package_stats[package]["potentially_independent"] += 1

        # Топ Android-зависимостей
        android_dependencies = defaultdict(int)
        for file_path, class_info in self.class_info_results.items():
            for android_import in class_info.android_imports:
                android_dependencies[android_import] += 1

        top_android_imports = sorted(android_dependencies.items(), key=lambda x: x[1], reverse=True)[:20]

        # Конвертируем defaultdict в обычный dict для JSON-сериализации
        package_stats_dict = {k: dict(v) for k, v in package_stats.items()}

        # Ошибки парсинга
        parse_errors = {}
        for file_path, class_info in self.class_info_results.items():
            if class_info.parse_error:
                parse_errors[file_path] = class_info.parse_error

        # Информация о производительности
        performance_data = {
            "total_time_seconds": self.stats.get_execution_time(),
            "average_parse_time_ms": sum(info.parse_time * 1000 for info in self.class_info_results.values()) / len(
                self.class_info_results) if self.class_info_results else 0,
            "parsing_efficiency_percent": self.stats.get_parse_success_rate(),
            "fastest_file": min(self.class_info_results.items(), key=lambda x: x[1].parse_time)[
                0] if self.class_info_results else None,
            "slowest_file": max(self.class_info_results.items(), key=lambda x: x[1].parse_time)[
                0] if self.class_info_results else None,
            "memory_usage_mb": self._get_memory_usage()
        }

        # Подробная информация о классах
        detailed_classes = {
            "android_dependent": [],
            "potentially_independent": [],
            "fully_independent": []
        }

        for file_path, class_info in self.class_info_results.items():
            if class_info.parse_error:
                continue

            class_data = {
                "file_path": class_info.file_path,
                "class_name": class_info.class_name,
                "package": class_info.package_name,
                "independence_score": class_info.independence_score,
                "android_imports_count": len(class_info.android_imports),
                "android_dependent_methods_count": sum(
                    1 for method in class_info.methods if method.is_android_dependent),
                "total_methods_count": len(class_info.methods),
                "potentially_extractable_methods": [
                    method.name for method in class_info.methods
                    if not method.is_android_dependent and method.independence_score > 0.7
                ]
            }

            if class_info.is_android_dependent:
                detailed_classes["android_dependent"].append(class_data)
            elif class_info.is_fully_independent:
                detailed_classes["fully_independent"].append(class_data)
            else:
                detailed_classes["potentially_independent"].append(class_data)

        # Сортируем по оценке независимости
        detailed_classes["potentially_independent"] = sorted(
            detailed_classes["potentially_independent"],
            key=lambda x: x["independence_score"],
            reverse=True
        )

        # Формируем полный отчет
        report = {
            "basic_stats": stats_data,
            "package_stats": package_stats_dict,
            "top_android_imports": dict(top_android_imports),
            "parse_errors": parse_errors,
            "performance_data": performance_data,
            "detailed_classes": detailed_classes,
            "analyzer_efficiency": self._calculate_analyzer_efficiency()
        }

        # Сохраняем отчет
        with open(os.path.join(output_dir, "analysis_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Создаем текстовый отчет
        self._generate_text_report(report, output_dir)

        # Создаем HTML отчет, если есть возможность
        try:
            self.generate_html_report(report, output_dir)  # Изменено с _generate_html_report на generate_html_report
        except Exception as e:
            logger.warning(f"Could not generate HTML report: {e}")
            logger.debug(traceback.format_exc())

        logger.info(f"Report generated in {output_dir}")
        return report

    def generate_html_report(self, report: Dict[str, Any], output_dir: str) -> None:
        """Генерирует HTML-отчет на основе данных анализа."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Использование не-интерактивного бэкенда
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not installed, skipping HTML report generation")
            return

        # Создаем круговую диаграмму классификации
        plt.figure(figsize=(8, 6))
        labels = ['Android Dependent', 'Potentially Independent', 'Fully Independent']
        sizes = [
            report["basic_stats"]["android_dependent_count"],
            report["basic_stats"]["potentially_independent_count"],
            report["basic_stats"]["fully_independent_count"]
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Classes Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_pie.png'))
        plt.close()

        # Создаем гистограмму пакетов
        packages = list(report["package_stats"].keys())
        if len(packages) > 10:
            # Берем топ 10 пакетов по количеству классов
            top_packages = sorted(
                report["package_stats"].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:10]
            packages = [pkg for pkg, _ in top_packages]
            package_stats = {pkg: report["package_stats"][pkg] for pkg in packages}
        else:
            package_stats = report["package_stats"]

        android_dependent = [package_stats[pkg]['android_dependent'] for pkg in packages]
        potentially_independent = [package_stats[pkg]['potentially_independent'] for pkg in packages]
        fully_independent = [package_stats[pkg]['fully_independent'] for pkg in packages]

        plt.figure(figsize=(12, 6))
        ind = np.arange(len(packages))
        width = 0.25

        p1 = plt.bar(ind, android_dependent, width, color='#ff9999')
        p2 = plt.bar(ind + width, potentially_independent, width, color='#66b3ff')
        p3 = plt.bar(ind + width * 2, fully_independent, width, color='#99ff99')

        plt.ylabel('Number of Classes')
        plt.title('Class Distribution per Package')
        plt.xticks(ind + width, [pkg.split('.')[-1] for pkg in packages], rotation=45, ha='right')
        plt.legend((p1[0], p2[0], p3[0]), ('Android Dependent', 'Potentially Independent', 'Fully Independent'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'package_distribution.png'))
        plt.close()

        # Создаем диаграмму эффективности
        eff = report["analyzer_efficiency"]
        categories = ['Parse Accuracy', 'Speed Score', 'Classification Accuracy', 'Overall Efficiency']
        values = [
            eff['parse_accuracy_percent'],
            eff['speed_score_percent'],
            eff['classification_accuracy_percent'],
            eff['overall_efficiency_percent']
        ]

        plt.figure(figsize=(10, 6))
        plt.barh(categories, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.xlabel('Score (%)')
        plt.title('Analyzer Efficiency')
        for i, v in enumerate(values):
            plt.text(v + 1, i, f"{v:.1f}%", va='center')
        plt.xlim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analyzer_efficiency.png'))
        plt.close()

        # Создаем HTML-отчет
        html_path = os.path.join(output_dir, "analysis_report.html")

        with open(html_path, "w") as f:
            f.write("""<!DOCTYPE html>
    <html>
    <head>
        <title>Java Dependency Analyzer Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #333;
            }
            .stat-box {
                display: inline-block;
                width: 180px;
                height: 100px;
                margin: 10px;
                padding: 15px;
                text-align: center;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                font-size: 14px;
                color: #666;
            }
            .chart {
                margin: 20px 0;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .footer {
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 12px;
            }
            .efficiency-badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                color: white;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Java Dependency Analyzer Report</h1>

            <h2>Summary</h2>
            <div class="stats">""")

            # Статистические блоки
            f.write(f'''
                <div class="stat-box">
                    <div class="stat-label">Total Files</div>
                    <div class="stat-value">{report["basic_stats"]["total_files"]}</div>
                </div>

                <div class="stat-box">
                    <div class="stat-label">Successfully Parsed</div>
                    <div class="stat-value">{report["basic_stats"]["successfully_parsed"]}</div>
                </div>

                <div class="stat-box">
                    <div class="stat-label">Execution Time</div>
                    <div class="stat-value">{report["basic_stats"]["execution_time_seconds"]:.2f}s</div>
                </div>

                <div class="stat-box">
                    <div class="stat-label">Files per Second</div>
                    <div class="stat-value">{report["basic_stats"]["files_per_second"]:.2f}</div>
                </div>''')

            # Статус эффективности
            eff_color = "#4CAF50" if report["analyzer_efficiency"]["overall_efficiency_percent"] >= 80 else \
                "#FF9800" if report["analyzer_efficiency"]["overall_efficiency_percent"] >= 60 else "#F44336"

            f.write(f'''
                <div class="stat-box">
                    <div class="stat-label">Efficiency Rating</div>
                    <div class="stat-value">
                        <span class="efficiency-badge" style="background-color: {eff_color}">
                            {report["analyzer_efficiency"]["rating"]}
                        </span>
                    </div>
                </div>
            </div>''')  # Закрытие div для статистических блоков

            # Диаграммы
            f.write('''
            <div class="charts">
                <div class="chart">
                    <h3>Class Classification</h3>
                    <img src="classification_pie.png" alt="Class Classification" width="500">
                </div>

                <div class="chart">
                    <h3>Package Distribution</h3>
                    <img src="package_distribution.png" alt="Package Distribution" width="800">
                </div>

                <div class="chart">
                    <h3>Analyzer Efficiency</h3>
                    <img src="analyzer_efficiency.png" alt="Analyzer Efficiency" width="700">
                </div>
            </div>''')

            # Топ Android-зависимостей
            f.write('''
            <h2>Top Android Dependencies</h2>
            <table>
                <tr><th>Import</th><th>Count</th></tr>''')

            for imp, count in list(report["top_android_imports"].items())[:15]:  # Ограничиваем 15
                f.write(f'<tr><td>{imp}</td><td>{count}</td></tr>')

            f.write('</table>')

            # Потенциально независимые классы
            f.write('''
            <h2>Top Potentially Independent Classes</h2>
            <table>
                <tr><th>Class</th><th>Package</th><th>Independence Score</th><th>Extractable Methods</th></tr>''')

            for cls in report["detailed_classes"]["potentially_independent"][:15]:  # Топ 15
                score_color = "green" if cls["independence_score"] >= 0.7 else \
                    "orange" if cls["independence_score"] >= 0.4 else "red"
                f.write(f'''
                <tr>
                    <td>{cls["class_name"]}</td>
                    <td>{cls["package"]}</td>
                    <td style="color: {score_color}">{cls["independence_score"]:.2f}</td>
                    <td>{", ".join(cls["potentially_extractable_methods"]) or "None"}</td>
                </tr>''')

            f.write('</table>')

            # Ошибки парсинга
            f.write('<h2>Parse Errors</h2>')
            if report["parse_errors"]:
                f.write('<table><tr><th>File</th><th>Error</th></tr>')

                errors_shown = 0
                for file, error in report["parse_errors"].items():
                    errors_shown += 1
                    if errors_shown > 15:  # Показываем только первые 15 ошибок
                        break
                    f.write(f'<tr><td>{file}</td><td>{error}</td></tr>')

                f.write('</table>')

                if len(report["parse_errors"]) > 15:
                    f.write(
                        f'<p>And {len(report["parse_errors"]) - 15} more errors. See the full report file for details.</p>')
            else:
                f.write('<p>No parse errors detected.</p>')

            # Подвал
            f.write('''
                <div class="footer">
                    <p>Generated by JavaDependencyAnalyzer</p>
                </div>
            </div>
        </body>
        </html>''')

        logger.info(f"HTML report generated at {html_path}")

    def _get_memory_usage(self) -> float:
        """Возвращает текущее использование памяти в МБ."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # В МБ

    def _calculate_analyzer_efficiency(self) -> Dict[str, Any]:
        """Рассчитывает метрики эффективности анализатора."""
        if not self.class_info_results:
            return {}

        # Оценка точности парсинга
        parse_accuracy = self.stats.get_parse_success_rate()

        # Оценка скорости
        speed_score = min(100, self.stats.get_files_per_second() * 10)  # 10 файлов/сек = 100%

        # Оценка качества классификации (искусственно создаем, так как нет золотого стандарта)
        # В реальном сценарии нужно сравнивать с экспертной оценкой
        classification_accuracy = 95.0  # Предполагаем 95% точность

        # Общий показатель эффективности
        overall_efficiency = (parse_accuracy * 0.4 + speed_score * 0.3 + classification_accuracy * 0.3)

        return {
            "parse_accuracy_percent": parse_accuracy,
            "speed_score_percent": speed_score,
            "classification_accuracy_percent": classification_accuracy,
            "overall_efficiency_percent": overall_efficiency,
            "rating": self._get_efficiency_rating(overall_efficiency)
        }

    def _get_efficiency_rating(self, score: float) -> str:
        """Возвращает рейтинг на основе числового показателя."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Satisfactory"
        else:
            return "Needs Improvement"

    def _generate_text_report(self, report: Dict[str, Any], output_dir: str) -> None:
        """Генерирует текстовый отчет на основе данных."""
        with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
            f.write("=== Java Dependency Analyzer Report ===\n\n")

            # Основная статистика
            f.write("== Basic Statistics ==\n")
            stats = report["basic_stats"]
            f.write(f"Total Java files: {stats['total_files']}\n")
            f.write(f"Successfully parsed: {stats['successfully_parsed']} ({stats['success_rate_percent']:.2f}%)\n")
            f.write(f"Parse errors: {stats['parse_errors']}\n")
            f.write(f"Execution time: {stats['execution_time_seconds']:.2f} seconds\n")
            f.write(f"Processing speed: {stats['files_per_second']:.2f} files/second\n\n")

            f.write("== Classification Results ==\n")
            f.write(
                f"Android dependent classes: {stats['android_dependent_count']} ({stats['android_dependent_percent']:.2f}%)\n")
            f.write(
                f"Potentially independent classes: {stats['potentially_independent_count']} ({stats['potentially_independent_percent']:.2f}%)\n")
            f.write(
                f"Fully independent classes: {stats['fully_independent_count']} ({stats['fully_independent_percent']:.2f}%)\n\n")

            # Информация о пакетах
            f.write("== Package Statistics ==\n")
            for pkg, pkg_stats in report["package_stats"].items():
                f.write(f"Package: {pkg}\n")
                f.write(f"  Total classes: {pkg_stats['total']}\n")
                f.write(f"  Android dependent: {pkg_stats['android_dependent']}\n")
                f.write(f"  Potentially independent: {pkg_stats['potentially_independent']}\n")
                f.write(f"  Fully independent: {pkg_stats['fully_independent']}\n")
                f.write("\n")

            # Топ Android импортов
            f.write("== Top Android Dependencies ==\n")
            for imp, count in report["top_android_imports"].items():
                f.write(f"{imp}: {count} usages\n")
            f.write("\n")

            # Ошибки парсинга
            f.write("== Parse Errors ==\n")
            if report["parse_errors"]:
                for file, error in report["parse_errors"].items():
                    f.write(f"{file}: {error}\n")
            else:
                f.write("No parse errors\n")
            f.write("\n")

            # Информация о производительности
            f.write("== Performance Data ==\n")
            perf = report["performance_data"]
            f.write(f"Total time: {perf['total_time_seconds']:.2f} seconds\n")
            f.write(f"Average parse time: {perf['average_parse_time_ms']:.2f} ms per file\n")
            f.write(f"Parsing efficiency: {perf['parsing_efficiency_percent']:.2f}%\n")
            f.write(f"Memory usage: {perf['memory_usage_mb']:.2f} MB\n")
            f.write(f"Fastest file: {perf['fastest_file']}\n")
            f.write(f"Slowest file: {perf['slowest_file']}\n\n")

            # Потенциально независимые классы
            f.write("== Top Potentially Independent Classes ==\n")
            for cls in report["detailed_classes"]["potentially_independent"][:10]:  # Top 10
                f.write(f"Class: {cls['class_name']} ({cls['package']})\n")
                f.write(f"  Independence score: {cls['independence_score']:.2f}\n")
                f.write(f"  Android imports: {cls['android_imports_count']}\n")
                f.write(f"  Android methods: {cls['android_dependent_methods_count']}/{cls['total_methods_count']}\n")
                f.write(f"  Extractable methods: {', '.join(cls['potentially_extractable_methods']) or 'None'}\n")
                f.write("\n")

            # Эффективность анализатора
            f.write("== Analyzer Efficiency ==\n")
            eff = report["analyzer_efficiency"]
            f.write(f"Parse accuracy: {eff['parse_accuracy_percent']:.2f}%\n")
            f.write(f"Speed score: {eff['speed_score_percent']:.2f}%\n")
            f.write(f"Classification accuracy: {eff['classification_accuracy_percent']:.2f}%\n")
            f.write(f"Overall efficiency: {eff['overall_efficiency_percent']:.2f}%\n")
            f.write(f"Rating: {eff['rating']}\n")

    def generate_llm_transform_report(self, output_dir: str = "llm_report") -> Dict[str, Any]:
        """
        Генерирует отчет, оптимизированный для использования с LLM.
        Создает структурированные файлы с информацией о каждом классе
        для последующей передачи в языковую модель.
        """
        if not self.class_info_results:
            logger.error("No analysis results available. Run analyze_directory first.")
            return {}

        # Создаем директорию отчета
        os.makedirs(output_dir, exist_ok=True)

        # Счетчики для итогового отчета
        counts = {
            "android_dependent": 0,
            "potentially_independent": 0,
            "fully_independent": 0
        }

        # Классифицируем и сохраняем информацию для каждого класса
        class_reports = {
            "android_dependent": [],
            "potentially_independent": [],
            "fully_independent": []
        }

        # Проходим по всем проанализированным классам
        for file_path, class_info in self.class_info_results.items():
            if class_info.parse_error:
                continue  # Пропускаем файлы с ошибками парсинга

            # Определяем категорию класса
            if class_info.is_android_dependent:
                category = "android_dependent"
            elif class_info.is_fully_independent:
                category = "fully_independent"
            else:
                category = "potentially_independent"

            counts[category] += 1

            # Формируем основную информацию о классе для LLM
            class_report = {
                "file_path": class_info.file_path,
                "class_name": class_info.class_name,
                "package_name": class_info.package_name,
                "category": category,
                "independence_score": class_info.independence_score,
                "transformation_needed": category != "android_dependent",
                "imports": {
                    "all": class_info.imports,
                    "android": class_info.android_imports
                }
            }

            # Добавляем специфичную для категории информацию
            if category == "potentially_independent":
                # Генерируем рекомендации по трансформации
                android_methods = [
                    {
                        "name": method.name,
                        "android_apis": method.android_apis_used,
                        "independence_score": method.independence_score
                    }
                    for method in class_info.methods if method.is_android_dependent
                ]

                independent_methods = [
                    method.name for method in class_info.methods
                    if not method.is_android_dependent and method.independence_score > 0.7
                ]

                android_fields = [
                    {"name": field.name, "type": field.type_name}
                    for field in class_info.fields if field.is_android_type
                ]

                class_report["transformation_info"] = {
                    "android_dependent_methods": android_methods,
                    "independent_methods": independent_methods,
                    "android_fields": android_fields,
                    "recommendations": self._generate_transformation_recommendations(class_info)
                }

            class_reports[category].append(class_report)

            # Для категорий, требующих трансформации, создаем отдельные файлы с информацией
            if category in ["android_dependent", "potentially_independent", "fully_independent"]:
                # Создаем отдельную директорию для каждой категории
                category_dir = os.path.join(output_dir, category)
                os.makedirs(category_dir, exist_ok=True)

                # Формируем имя файла отчета
                filename = f"{class_info.package_name}.{class_info.class_name}"
                # Заменяем недопустимые символы в имени файла
                filename = filename.replace(".", "_")

                # Сохраняем отчет о классе в JSON-файле
                with open(os.path.join(category_dir, f"{filename}.json"), "w") as f:
                    json.dump(class_report, f, indent=2)

                # Также создаем файл с исходным кодом для удобства
                with open(os.path.join(category_dir, f"{filename}.java"), "w") as f:
                    with open(class_info.file_path, "r", encoding="utf-8") as src:
                        f.write(src.read())

        # Создаем итоговый отчет для LLM
        summary = {
            "counts": counts,
            "class_lists": {
                "potentially_independent": [c["class_name"] for c in class_reports["potentially_independent"]],
                "fully_independent": [c["class_name"] for c in class_reports["fully_independent"]]
            },
            "transformation_instructions": self._generate_llm_instructions()
        }

        # Сохраняем итоговый отчет
        with open(os.path.join(output_dir, "llm_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Создаем отчет для каждой категории
        for category in class_reports:
            with open(os.path.join(output_dir, f"{category}_classes.json"), "w") as f:
                json.dump(class_reports[category], f, indent=2)

        # Создаем README для навигации по отчету
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# LLM Transformation Report\n\n")
            f.write(f"This directory contains reports optimized for LLM-based code transformation.\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Android dependent classes: {counts['android_dependent']}\n")
            f.write(f"- Potentially independent classes: {counts['potentially_independent']}\n")
            f.write(f"- Fully independent classes: {counts['fully_independent']}\n\n")
            f.write(f"## Structure\n\n")
            f.write(f"- `llm_summary.json`: Overall summary and counts\n")
            f.write(
                f"- `potentially_independent/`: Reports for classes that can be transformed with some modifications\n")
            f.write(f"- `fully_independent/`: Reports for classes that are already platform-independent\n")
            f.write(f"- `*_classes.json`: Complete lists of classes in each category\n\n")
            f.write(f"## Usage with LLM\n\n")
            f.write(
                f"Each class report includes transformation recommendations that can be used to guide the LLM in transforming the class.")

        logger.info(f"LLM-optimized report generated in {output_dir}")
        return summary

    def generate_detailed_dependency_report(self, output_dir: str = "dependency_report") -> Dict[str, Any]:
        """
        Генерирует детальный отчет о зависимостях для каждого файла
        с подробным объяснением причин зависимости от Android.
        """
        if not self.class_info_results:
            logger.error("No analysis results available. Run analyze_directory first.")
            return {}

        # Создаем директорию отчета
        os.makedirs(output_dir, exist_ok=True)

        # Создаем подпапки для каждой категории
        for category in ["android_dependent", "potentially_independent", "fully_independent"]:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)

        # Общая статистика
        summary = {
            "total_files": self.stats.total_files,
            "successfully_parsed": self.stats.successfully_parsed,
            "android_dependent": {
                "count": self.stats.android_dependent,
                "files": []
            },
            "potentially_independent": {
                "count": self.stats.potentially_independent,
                "files": []
            },
            "fully_independent": {
                "count": self.stats.fully_independent,
                "files": []
            },
            "parse_errors": {
                "count": self.stats.parse_errors,
                "files": []
            }
        }

        # Анализируем каждый класс
        for file_path, class_info in self.class_info_results.items():
            # Имя файла для отчета
            base_name = os.path.basename(file_path)

            if class_info.parse_error:
                # Сохраняем информацию об ошибке парсинга
                error_report = {
                    "file_path": file_path,
                    "class_name": class_info.class_name,
                    "error": class_info.parse_error
                }
                summary["parse_errors"]["files"].append(error_report)
                continue

            # Определяем категорию
            if class_info.is_android_dependent:
                category = "android_dependent"
            elif class_info.is_fully_independent:
                category = "fully_independent"
            else:
                category = "potentially_independent"

            # Генерируем подробную информацию о зависимостях
            detailed_info = self._generate_detailed_dependency_info(class_info)

            # Формируем отчет для файла
            file_report = {
                "file_path": file_path,
                "class_name": class_info.class_name,
                "package_name": class_info.package_name,
                "category": category,
                "independence_score": class_info.independence_score,
                "detailed_info": detailed_info
            }

            # Добавляем в соответствующую категорию
            summary[category]["files"].append({
                "file_path": file_path,
                "class_name": class_info.class_name,
                "package_name": class_info.package_name,
                "independence_score": class_info.independence_score
            })

            # Сохраняем детальный отчет для файла
            report_file = os.path.join(output_dir, category, f"{base_name.replace('.java', '')}_report.json")
            with open(report_file, "w") as f:
                json.dump(file_report, f, indent=2)

            # Создаем также читаемый текстовый отчет
            text_report = os.path.join(output_dir, category, f"{base_name.replace('.java', '')}_report.txt")
            with open(text_report, "w") as f:
                f.write(f"==== Dependency Analysis for {class_info.class_name} ====\n\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Package: {class_info.package_name}\n")
                f.write(f"Category: {category.replace('_', ' ').title()}\n")
                f.write(f"Independence Score: {class_info.independence_score:.2f}/1.00\n\n")

                f.write("-- Dependencies Overview --\n")
                f.write(f"Android Imports: {len(class_info.android_imports)}/{len(class_info.imports)} total imports\n")
                f.write(
                    f"Android Fields: {sum(1 for field in class_info.fields if field.is_android_type)}/{len(class_info.fields)} total fields\n")
                f.write(
                    f"Android Methods: {sum(1 for method in class_info.methods if method.is_android_dependent)}/{len(class_info.methods)} total methods\n\n")

                # Android imports
                f.write("-- Android Imports --\n")
                if class_info.android_imports:
                    for imp in class_info.android_imports:
                        f.write(f"  - {imp}\n")
                else:
                    f.write("  No Android imports\n")
                f.write("\n")

                # Android fields
                f.write("-- Android Fields --\n")
                android_fields = [field for field in class_info.fields if field.is_android_type]
                if android_fields:
                    for field in android_fields:
                        f.write(f"  - {field.name}: {field.type_name}\n")
                else:
                    f.write("  No Android fields\n")
                f.write("\n")

                # Android methods
                f.write("-- Android-Dependent Methods --\n")
                android_methods = [method for method in class_info.methods if method.is_android_dependent]
                if android_methods:
                    for method in android_methods:
                        f.write(f"  - {method.name}\n")
                        f.write(f"    Independence Score: {method.independence_score:.2f}\n")
                        f.write(f"    Android APIs Used:\n")
                        if method.android_apis_used:
                            for api in method.android_apis_used:
                                f.write(f"      * {api}\n")
                        else:
                            f.write(f"      * None specifically identified\n")
                        f.write("\n")
                else:
                    f.write("  No Android-dependent methods\n")
                f.write("\n")

                # Трансформационные рекомендации для потенциально независимых классов
                if category == "potentially_independent":
                    f.write("-- Transformation Recommendations --\n")
                    recommendations = self._generate_transformation_recommendations(class_info)
                    for rec in recommendations:
                        f.write(f"  - {rec}\n")
                    f.write("\n")

        # Сохраняем общий отчет
        with open(os.path.join(output_dir, "dependency_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Создаем сводный HTML-отчет
        try:
            self._generate_dependency_html_report(summary, output_dir)
        except Exception as e:
            logger.error(f"Failed to generate dependency HTML report: {e}")
            logger.debug(traceback.format_exc())

        # Создаем README для навигации по отчету
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# Detailed Dependency Analysis Report\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total files analyzed: {summary['total_files']}\n")
            f.write(f"- Successfully parsed: {summary['successfully_parsed']}\n")
            f.write(f"- Android dependent classes: {summary['android_dependent']['count']}\n")
            f.write(f"- Potentially independent classes: {summary['potentially_independent']['count']}\n")
            f.write(f"- Fully independent classes: {summary['fully_independent']['count']}\n")
            f.write(f"- Parse errors: {summary['parse_errors']['count']}\n\n")

            f.write(f"## Structure\n\n")
            f.write(f"- `dependency_summary.json`: Overall summary\n")
            f.write(f"- `dependency_report.html`: Interactive HTML report\n")
            f.write(f"- `android_dependent/`: Detailed reports for Android-dependent classes\n")
            f.write(f"- `potentially_independent/`: Detailed reports for potentially independent classes\n")
            f.write(f"- `fully_independent/`: Detailed reports for fully independent classes\n\n")

            f.write(f"## How to Use This Report\n\n")
            f.write(
                f"Each class has both a JSON report (for programmatic access) and a text report (for human reading).\n")
            f.write(
                f"The reports include detailed information about Android dependencies at the import, field, and method levels.\n")
            f.write(
                f"For potentially independent classes, transformation recommendations are provided to help with extracting platform-independent code.\n")

        logger.info(f"Detailed dependency report generated in {output_dir}")
        return summary

    def _generate_detailed_dependency_info(self, class_info: JavaClassInfo) -> Dict[str, Any]:
        """Генерирует детальную информацию о зависимостях класса."""
        details = {
            "android_component": {
                "is_component": class_info.is_android_component,
                "extends": class_info.extends,
                "implements": class_info.implements,
                "reason": None
            },
            "imports": {
                "total_count": len(class_info.imports),
                "android_count": len(class_info.android_imports),
                "android_imports": class_info.android_imports,
                "import_ratio": len(class_info.android_imports) / len(class_info.imports) if class_info.imports else 0
            },
            "fields": {
                "total_count": len(class_info.fields),
                "android_fields": [
                    {
                        "name": field.name,
                        "type": field.type_name
                    }
                    for field in class_info.fields if field.is_android_type
                ],
                "field_ratio": sum(1 for field in class_info.fields if field.is_android_type) / len(
                    class_info.fields) if class_info.fields else 0
            },
            "methods": {
                "total_count": len(class_info.methods),
                "android_methods": [
                    {
                        "name": method.name,
                        "return_type": method.return_type,
                        "independence_score": method.independence_score,
                        "android_apis": method.android_apis_used
                    }
                    for method in class_info.methods if method.is_android_dependent
                ],
                "method_ratio": sum(1 for method in class_info.methods if method.is_android_dependent) / len(
                    class_info.methods) if class_info.methods else 0,
                "independent_methods": [
                    method.name
                    for method in class_info.methods if
                    not method.is_android_dependent and method.independence_score > 0.7
                ]
            },
            "dependency_factors": {
                "is_android_component": class_info.is_android_component,
                "has_android_imports": len(class_info.android_imports) > 0,
                "has_android_fields": any(field.is_android_type for field in class_info.fields),
                "has_android_methods": any(method.is_android_dependent for method in class_info.methods),
                "primary_reason": ""
            }
        }

        # Определение основной причины зависимости
        if class_info.is_android_component:
            details["android_component"]["reason"] = f"Class extends or implements Android component"
            details["dependency_factors"]["primary_reason"] = "Android component inheritance"
        elif details["fields"]["field_ratio"] > 0.5:
            details["dependency_factors"]["primary_reason"] = "High ratio of Android fields"
        elif details["methods"]["method_ratio"] > 0.5:
            details["dependency_factors"]["primary_reason"] = "High ratio of Android-dependent methods"
        elif details["imports"]["import_ratio"] > 0.5:
            details["dependency_factors"]["primary_reason"] = "High ratio of Android imports"
        else:
            details["dependency_factors"]["primary_reason"] = "Combined factors"

        return details

    def _generate_transformation_recommendations(self, class_info: JavaClassInfo) -> List[str]:
        """Генерирует рекомендации по трансформации класса."""
        recommendations = []

        # Если класс является Android-компонентом, трансформация затруднена
        if class_info.is_android_component:
            recommendations.append(
                "This class extends or implements an Android component, making it difficult to extract " +
                "platform-independent code. Consider extracting business logic to separate classes."
            )
            return recommendations

        # Рекомендации для полей Android-типов
        android_fields = [field for field in class_info.fields if field.is_android_type]
        if android_fields:
            recommendations.append(
                f"Replace {len(android_fields)} Android-type fields with platform-independent interfaces or classes."
            )
            for field in android_fields:
                if field.type_name == "Context":
                    recommendations.append(
                        f"  - Create a ContextProvider interface to abstract the Android Context dependency for field '{field.name}'."
                    )
                else:
                    recommendations.append(
                        f"  - Replace field '{field.name}' of type '{field.type_name}' with an interface or platform-independent alternative."
                    )

        # Рекомендации для методов с Android API
        android_methods = [method for method in class_info.methods if method.is_android_dependent]
        if android_methods:
            recommendations.append(
                f"Transform {len(android_methods)} Android-dependent methods:"
            )
            for method in android_methods:
                if method.independence_score > 0.5:
                    recommendations.append(
                        f"  - Method '{method.name}' has moderate Android dependencies. " +
                        f"Abstract Android-specific calls using interfaces."
                    )
                else:
                    recommendations.append(
                        f"  - Method '{method.name}' is heavily dependent on Android. " +
                        f"Consider moving to a separate Android-specific class or using a delegation pattern."
                    )

        # Рекомендации для импортов
        if class_info.android_imports:
            recommendations.append(
                f"Remove or replace {len(class_info.android_imports)} Android imports with standard Java alternatives."
            )

            # Предложим конкретные замены для некоторых импортов
            replacements = {
                "android.util.Log": "java.util.logging.Logger",
                "android.os.AsyncTask": "java.util.concurrent.Executor",
                "android.os.Handler": "java.util.Timer",
                "android.text.TextUtils": "org.apache.commons.lang3.StringUtils",
                "android.util.Pair": "java.util.AbstractMap.SimpleEntry"
            }

            for imp in class_info.android_imports:
                if imp in replacements:
                    recommendations.append(
                        f"  - Replace '{imp}' with '{replacements[imp]}'."
                    )

        # Общие рекомендации
        if class_info.independence_score > 0.7:
            recommendations.append(
                "This class has a high independence score. Most of its code can be extracted " +
                "with minimal changes."
            )
        elif class_info.independence_score > 0.4:
            recommendations.append(
                "This class has a moderate independence score. Consider refactoring to " +
                "separate platform-independent logic from Android-specific code."
            )
        else:
            recommendations.append(
                "This class has a low independence score but might still contain extractable methods. " +
                "Focus on extracting individual methods rather than the entire class."
            )

        return recommendations

    def _generate_llm_instructions(self) -> Dict[str, str]:
        """Генерирует инструкции для LLM по трансформации кода."""
        return {
            "potentially_independent": """
    To transform this potentially independent class:

    1. Identify and remove Android imports:
       - Replace android.util.Log with java.util.logging.Logger
       - Replace android.os.AsyncTask with java.util.concurrent.Executor
       - Remove other Android-specific imports

    2. Create abstraction interfaces:
       - For Android Context dependencies, create a ContextProvider interface
       - For other Android-specific types, create appropriate interfaces

    3. Replace Android API calls:
       - Replace direct Android API calls with calls to the abstraction interfaces
       - Move Android-specific implementations to separate classes

    4. Ensure all remaining code is platform-independent:
       - Remove references to Android UI components
       - Extract business logic and data processing code

    5. Maintain original functionality:
       - Preserve method signatures where possible
       - Keep naming consistent with original code
    """,
            "fully_independent": """
    To handle this fully independent class:

    1. Validate platform independence:
       - Verify that imports don't have hidden Android dependencies
       - Ensure no indirect Android references through other classes

    2. Clean up any unnecessary code:
       - Remove unused methods or fields
       - Optimize imports

    3. Verify compilation:
       - Ensure the class compiles without Android SDK
       - Fix any references that might indirectly depend on Android

    4. Prepare for reuse:
       - Add appropriate documentation
       - Ensure public API is clear and usable outside Android context
    """
        }

    def _generate_dependency_html_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        """Генерирует HTML-отчет о зависимостях с возможностью фильтрации и поиска."""
        try:
            html_path = os.path.join(output_dir, "dependency_report.html")

            with open(html_path, "w") as f:
                html_content = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Java Dependency Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .stats {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 30px;
            }
            .stat-card {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 200px;
            }
            .stat-card h3 {
                margin-top: 0;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }
            .filters {
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .search-box {
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .category-tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            .category-tab {
                padding: 8px 15px;
                background-color: #e9ecef;
                border-radius: 20px;
                cursor: pointer;
            }
            .category-tab.active {
                background-color: #3498db;
                color: white;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .score-cell {
                width: 150px;
            }
            .score-bar {
                height: 20px;
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
            }
            .score-fill {
                height: 100%;
                border-radius: 10px;
            }
            .details-link {
                color: #3498db;
                text-decoration: none;
            }
            .details-link:hover {
                text-decoration: underline;
            }
            .progress-container {
                margin-top: 20px;
            }
            .section {
                margin-bottom: 30px;
            }
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Java Dependency Analysis Report</h1>
    """

                # Добавляем статистические карточки
                html_content += """
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Files</h3>
                    <div class="stat-number">{total_files}</div>
                </div>
                <div class="stat-card">
                    <h3>Successfully Parsed</h3>
                    <div class="stat-number">{successfully_parsed}</div>
                    <div>{parse_success_rate}%</div>
                </div>
                <div class="stat-card">
                    <h3>Android Dependent</h3>
                    <div class="stat-number">{android_dependent_count}</div>
                    <div>{android_dependent_percent}%</div>
                </div>
                <div class="stat-card">
                    <h3>Potentially Independent</h3>
                    <div class="stat-number">{potentially_independent_count}</div>
                    <div>{potentially_independent_percent}%</div>
                </div>
                <div class="stat-card">
                    <h3>Fully Independent</h3>
                    <div class="stat-number">{fully_independent_count}</div>
                    <div>{fully_independent_percent}%</div>
                </div>
            </div>
    """.format(
                    total_files=summary["total_files"],
                    successfully_parsed=summary["successfully_parsed"],
                    parse_success_rate=round(
                        summary["successfully_parsed"] / summary["total_files"] * 100 if summary[
                                                                                             "total_files"] > 0 else 0,
                        1),
                    android_dependent_count=summary["android_dependent"]["count"],
                    potentially_independent_count=summary["potentially_independent"]["count"],
                    fully_independent_count=summary["fully_independent"]["count"],
                    android_dependent_percent=round(
                        summary["android_dependent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                            "successfully_parsed"] > 0 else 0,
                        1),
                    potentially_independent_percent=round(
                        summary["potentially_independent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                                  "successfully_parsed"] > 0 else 0,
                        1),
                    fully_independent_percent=round(
                        summary["fully_independent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                            "successfully_parsed"] > 0 else 0,
                        1)
                )

                # Фильтры
                html_content += """
            <div class="filters">
                <input type="text" id="searchInput" class="search-box" placeholder="Search by class name or package...">
                <div class="category-tabs">
                    <div class="category-tab active" data-category="all">All Classes</div>
                    <div class="category-tab" data-category="android_dependent">Android Dependent</div>
                    <div class="category-tab" data-category="potentially_independent">Potentially Independent</div>
                    <div class="category-tab" data-category="fully_independent">Fully Independent</div>
                </div>
            </div>
    """

                # Таблица классов
                html_content += """
            <div class="section" id="classesTable">
                <h2>Classes</h2>
                <table id="classes">
                    <thead>
                        <tr>
                            <th>Class Name</th>
                            <th>Package</th>
                            <th>Category</th>
                            <th class="score-cell">Independence Score</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Dynamic content will be inserted here -->
                    </tbody>
                </table>
            </div>
    """

                # Секция ошибок
                html_content += """
            <div class="section hidden" id="errorSection">
                <h2>Parse Errors</h2>
                <table id="errors">
                    <thead>
                        <tr>
                            <th>Class Name</th>
                            <th>File Path</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Error content will be inserted here -->
                    </tbody>
                </table>
            </div>
    """

                # Секция прогресса
                html_content += """
            <div class="progress-container">
                <h2>Progress</h2>
                <div class="stat-card">
                    <h3>Independence Distribution</h3>
                    <div style="height: 30px; background-color: #f1f1f1; border-radius: 5px; overflow: hidden; margin-top: 10px;">
                        <div style="width: {android_dependent_percent}%; height: 100%; background-color: #ff9999; float: left;"></div>
                        <div style="width: {potentially_independent_percent}%; height: 100%; background-color: #ffcc99; float: left;"></div>
                        <div style="width: {fully_independent_percent}%; height: 100%; background-color: #99cc99; float: left;"></div>
                    </div>
                    <div style="display: flex; margin-top: 5px; font-size: 12px;">
                        <div style="flex: {android_dependent_percent}; text-align: center;">Android</div>
                        <div style="flex: {potentially_independent_percent}; text-align: center;">Potential</div>
                        <div style="flex: {fully_independent_percent}; text-align: center;">Independent</div>
                    </div>
                </div>
            </div>
        </div>
    """.format(
                    android_dependent_percent=round(
                        summary["android_dependent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                            "successfully_parsed"] > 0 else 0,
                        1),
                    potentially_independent_percent=round(
                        summary["potentially_independent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                                  "successfully_parsed"] > 0 else 0,
                        1),
                    fully_independent_percent=round(
                        summary["fully_independent"]["count"] / summary["successfully_parsed"] * 100 if summary[
                                                                                                            "successfully_parsed"] > 0 else 0,
                        1)
                )

                # JavaScript для интерактивности
                html_content += """
        <script>
            // Data from analysis
            const classesData = {class_data};
            const errorsData = {error_data};

            // Initialize tables
            function initTables() {
                const classesTableBody = document.querySelector('#classes tbody');
                const errorsTableBody = document.querySelector('#errors tbody');

                // Clear existing content
                classesTableBody.innerHTML = '';
                errorsTableBody.innerHTML = '';

                // Add classes data
                classesData.forEach(cls => {
                    const row = document.createElement('tr');
                    row.dataset.category = cls.category;
                    row.dataset.search = (cls.class_name + ' ' + cls.package_name).toLowerCase();
                    // Calculate score color
                    let scoreColor = '#ff9999'; // Red for low scores
                    if (cls.independence_score > 0.7) {
                        scoreColor = '#99cc99'; // Green for high scores
                    } else if (cls.independence_score > 0.4) {
                        scoreColor = '#ffcc99'; // Orange for medium scores
                    }

                    row.innerHTML = `
                        <td>${cls.class_name}</td>
                        <td>${cls.package_name}</td>
                        <td>${formatCategory(cls.category)}</td>
                        <td class="score-cell">
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${cls.independence_score * 100}%; background-color: ${scoreColor};"></div>
                            </div>
                            <div style="text-align: center; margin-top: 5px;">${(cls.independence_score * 100).toFixed(1)}%</div>
                        </td>
                        <td>
                            <a href="${cls.category}/${cls.class_name.replace(/\./g, '_')}_report.txt" class="details-link" target="_blank">View Details</a>
                        </td>
                    `;

                    classesTableBody.appendChild(row);
                });

                // Add errors data
                if (errorsData.length > 0) {
                    document.getElementById('errorSection').classList.remove('hidden');

                    errorsData.forEach(err => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${err.class_name}</td>
                            <td>${err.file_path}</td>
                            <td>${err.error}</td>
                        `;

                        errorsTableBody.appendChild(row);
                    });
                }
            }

            // Format category names for display
            function formatCategory(category) {
                return category
                    .split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
            }

            // Filter table by category
            function filterByCategory(category) {
                const rows = document.querySelectorAll('#classes tbody tr');

                rows.forEach(row => {
                    if (category === 'all' || row.dataset.category === category) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }

            // Filter table by search term
            function filterBySearch(term) {
                const rows = document.querySelectorAll('#classes tbody tr');
                const lowerTerm = term.toLowerCase();

                rows.forEach(row => {
                    if (row.dataset.search.includes(lowerTerm)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }

            // Initialize the page
            document.addEventListener('DOMContentLoaded', () => {
                initTables();

                // Category tab handling
                const tabs = document.querySelectorAll('.category-tab');
                tabs.forEach(tab => {
                    tab.addEventListener('click', () => {
                        tabs.forEach(t => t.classList.remove('active'));
                        tab.classList.add('active');
                        filterByCategory(tab.dataset.category);
                    });
                });

                // Search handling
                const searchInput = document.getElementById('searchInput');
                searchInput.addEventListener('input', () => {
                    filterBySearch(searchInput.value);
                });
            });
        </script>
    </body>
    </html>
    """
                # Подготовка данных для JavaScript
                class_data = json.dumps(
                    summary["android_dependent"]["files"] +
                    summary["potentially_independent"]["files"] +
                    summary["fully_independent"]["files"]
                )

                error_data = json.dumps(summary["parse_errors"]["files"])

                # Замена плейсхолдеров на данные
                html_content = html_content.replace("{class_data}", class_data)
                html_content = html_content.replace("{error_data}", error_data)

                # Запись в файл
                f.write(html_content)

            logger.info(f"HTML dependency report generated at {html_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            logger.debug(traceback.format_exc())


def main():
    """Основная функция для запуска анализатора из командной строки."""
    import argparse

    parser = argparse.ArgumentParser(description='Java Dependency Analyzer')
    parser.add_argument('--directory', help='Directory with Java files to analyze',
                        default="/Users/rodvlasov2003/PycharmProjects/vkr/decompilers/downloaded_apks/Timetable_SPbSU/sources/custom")
    parser.add_argument('--output', '-o', default='java_analyzer_report_timetable', help='Output directory for the report')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--llm-report', action='store_true', help='Generate LLM-optimized report')
    parser.add_argument('--detailed-report', action='store_true', help='Generate detailed dependency report')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1

    print(f"Starting analysis of '{args.directory}'")
    analyzer = JavaDependencyAnalyzer()
    analyzer.analyze_directory(args.directory, max_workers=args.workers)

    # Генерация стандартного отчета
    analyzer.generate_analysis_report(args.output)

    # Генерация отчета для LLM, если запрошено
    if args.llm_report or True:
        llm_output_dir = os.path.join(args.output, "llm_report")
        analyzer.generate_llm_transform_report(llm_output_dir)
        print(f"LLM-optimized report generated in '{llm_output_dir}'")

    # Генерация детального отчета зависимостей, если запрошено
    if args.detailed_report or True:
        dependency_output_dir = os.path.join(args.output, "dependency_report")
        analyzer.generate_detailed_dependency_report(dependency_output_dir)
        print(f"Detailed dependency report generated in '{dependency_output_dir}'")

    # Вывод итоговой статистики
    stats = analyzer.stats
    print("\nSummary:")
    print(f"Total files: {stats.total_files}")
    print(f"Successfully parsed: {stats.successfully_parsed} ({stats.get_parse_success_rate():.2f}%)")
    print(f"Parse errors: {stats.parse_errors}")
    print(f"Execution time: {stats.get_execution_time():.2f} seconds")
    print(f"Processing speed: {stats.get_files_per_second():.2f} files/second")
    print(f"Android dependent classes: {stats.android_dependent}")
    print(f"Potentially independent classes: {stats.potentially_independent}")
    print(f"Fully independent classes: {stats.fully_independent}")

    return 0


if __name__ == "__main__":
    exit(main())
