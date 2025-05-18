import os
from kpi import KPI
from dotenv import load_dotenv

load_dotenv()

kpi = KPI()

output_dir = os.getenv("OUTPUT_DIR", "llm/llm_output_timetable")
original_source_dir = os.getenv("SOURCE_DIR", "decompilers/downloaded_apks/Timetable_SPbSU/sources/custom")
analysis_report_path = os.getenv("ANALYSIS_REPORT", "java_analyzer_report_timetable/analysis_report.json")

all_kpi = kpi.calculate_all_kpi(
    output_dir=output_dir,
    original_source_dir=original_source_dir,
    analysis_report_path=analysis_report_path,
)

print("\n## Сводная таблица KPI\n")
print("| KPI | Значение | Целевое значение | Соответствие |")
print("|-----|----------|------------------|--------------|")
print(f"| Полнота извлечения | {all_kpi['extraction_completeness']:.2f}% | >85% | {'✅' if all_kpi['extraction_completeness'] >= 85 else '❌'} |")
print(f"| Точность извлечения | {all_kpi['extraction_accuracy']:.2f}% | >75% | {'✅' if all_kpi['extraction_accuracy'] >= 75 else '❌'} |")
print(f"| Удаление зависимостей | {all_kpi['dependency_removal_rate']:.2f}% | >95% | {'✅' if all_kpi['dependency_removal_rate'] >= 95 else '❌'} |")
print(f"| Компилируемость | {all_kpi['compilability']:.2f}% | >95% | {'✅' if all_kpi['compilability'] >= 95 else '❌'} |")
print(f"| Скорость (файлов/мин) | {all_kpi['speed_metrics']['files_per_minute']:.2f} | >0.5 | {'✅' if all_kpi['speed_metrics']['files_per_minute'] >= 0.5 else '❌'} |")
print(f"| Токены на файл | {all_kpi['token_metrics']['average_tokens_per_file']:.2f} | <10000 | {'✅' if all_kpi['token_metrics']['average_tokens_per_file'] < 20000 else '❌'} |")
print(f"| Сохранение структуры | {all_kpi['structure_preservation']:.2f}% | >75% | {'✅' if all_kpi['structure_preservation'] >= 75 else '❌'} |")
print(f"| Коэффициент размера | {all_kpi['size_metrics']['total_size_ratio']:.2f}x | 0.3-1.1 | {'✅' if 0.3 <= all_kpi['size_metrics']['total_size_ratio'] <= 1.1 else '❌'} |")
