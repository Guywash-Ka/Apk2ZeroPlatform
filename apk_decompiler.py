#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import platform
import shutil
import time
from datetime import datetime


class ApkDecompiler:

    def __init__(self, jadx_path=None, output_dir="decompiled_output"):
        self.jadx_path = jadx_path
        self.output_dir = output_dir
        self.stats = {
            "start_time": None,
            "end_time": None,
            "files_processed": 0,
            "classes_count": 0,
            "errors": []
        }

    def find_jadx(self):
        if self.jadx_path and os.path.exists(self.jadx_path):
            return True

        jadx_binary = "jadx.bat" if platform.system() == "Windows" else "jadx"

        if shutil.which(jadx_binary):
            self.jadx_path = jadx_binary
            return True

        common_paths = []

        if platform.system() == "Windows":
            common_paths = [
                os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "jadx", "bin", "jadx.bat"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "jadx", "bin", "jadx.bat"),
                os.path.join(os.environ.get("LOCALAPPDATA", "C:\\Users\\User\\AppData\\Local"), "jadx", "bin",
                             "jadx.bat")
            ]
        elif platform.system() == "Linux":
            common_paths = [
                "/usr/bin/jadx",
                "/usr/local/bin/jadx",
                os.path.expanduser("~/jadx/bin/jadx")
            ]
        elif platform.system() == "Darwin":
            common_paths = [
                "/usr/local/bin/jadx",
                "/opt/homebrew/bin/jadx",
                os.path.expanduser("~/jadx/bin/jadx")
            ]

        for path in common_paths:
            if os.path.exists(path):
                self.jadx_path = path
                return True

        return False

    def decompile_apk(self, apk_path, options=None):
        if not os.path.exists(apk_path):
            print(f"Ошибка: APK файл не найден: {apk_path}")
            return False

        if not self.find_jadx():
            print("Ошибка: JADX не найден. Пожалуйста, установите JADX или укажите путь к нему.")
            print("Инструкция по установке JADX: https://github.com/skylot/jadx#installation")
            return False

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        apk_name = os.path.basename(apk_path).replace(".apk", "")
        output_path = os.path.join(self.output_dir, apk_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cmd = [self.jadx_path, "-d", output_path]

        if options:
            cmd.extend(options)

        cmd.append(apk_path)

        print(f"Запуск декомпиляции: {' '.join(cmd)}")
        self.stats["start_time"] = time.time()

        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode == 0:
                print(f"Декомпиляция успешно завершена. Результат сохранен в: {output_path}")
                self.stats["files_processed"] += 1

                java_files_count = 0
                for root, _, files in os.walk(os.path.join(output_path, "sources")):
                    for file in files:
                        if file.endswith(".java"):
                            java_files_count += 1

                self.stats["classes_count"] = java_files_count
                print(f"Декомпилировано классов: {java_files_count}")

                self._generate_report(apk_path, output_path, process.stdout)

                return True
            else:
                print(f"Ошибка при декомпиляции: {process.stdout}")
                self.stats["errors"].append(process.stderr)
                return False

        except Exception as e:
            print(f"Исключение при декомпиляции: {str(e)}")
            self.stats["errors"].append(str(e))
            return False
        finally:
            self.stats["end_time"] = time.time()

    def _generate_report(self, apk_path, output_path, jadx_output):
        report_path = os.path.join(output_path, "decompilation_report.txt")

        try:
            with open(report_path, "w") as f:
                f.write("==== Отчет о декомпиляции APK файла ====\n\n")
                f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"APK файл: {os.path.abspath(apk_path)}\n")
                f.write(f"Выходная директория: {os.path.abspath(output_path)}\n")
                f.write(f"Инструмент: JADX ({self.jadx_path})\n\n")

                f.write("--- Статистика ---\n")
                f.write(f"Время декомпиляции: {self.stats['end_time'] - self.stats['start_time']:.2f} секунд\n")
                f.write(f"Количество декомпилированных классов: {self.stats['classes_count']}\n\n")

                f.write("--- Вывод JADX ---\n")
                f.write(jadx_output)

                if self.stats["errors"]:
                    f.write("\n--- Ошибки ---\n")
                    for error in self.stats["errors"]:
                        f.write(f"{error}\n")

            print(f"Отчет о декомпиляции сохранен в: {report_path}")

        except Exception as e:
            print(f"Ошибка при создании отчета: {str(e)}")

    def batch_decompile(self, apk_dir, options=None):
        if not os.path.exists(apk_dir):
            print(f"Ошибка: Директория не найдена: {apk_dir}")
            return 0

        apk_files = [f for f in os.listdir(apk_dir) if f.endswith(".apk")]

        if not apk_files:
            print(f"В директории {apk_dir} не найдено APK файлов.")
            return 0

        print(f"Найдено {len(apk_files)} APK файлов для декомпиляции.")

        success_count = 0
        for i, apk_file in enumerate(apk_files, 1):
            print(f"\n[{i}/{len(apk_files)}] Декомпиляция {apk_file}...")
            full_path = os.path.join(apk_dir, apk_file)
            if self.decompile_apk(full_path, options):
                success_count += 1

        print(f"\nИтого: успешно декомпилировано {success_count} из {len(apk_files)} APK файлов.")
        return success_count


def main():
    parser = argparse.ArgumentParser(description="Декомпиляция APK файлов с помощью JADX")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Путь к APK файлу для декомпиляции")
    group.add_argument("-d", "--directory", help="Путь к директории с APK файлами для пакетной декомпиляции")

    parser.add_argument("-o", "--output", default="decompiled_output",
                        help="Директория для вывода результатов (по умолчанию: decompiled_output)")
    parser.add_argument("-j", "--jadx", help="Путь к исполняемому файлу JADX")
    parser.add_argument("--show-bad-code", action="store_true",
                        help="Включить вывод некорректного кода")
    parser.add_argument("--no-imports", action="store_true",
                        help="Отключить вывод import-ов")
    parser.add_argument("--no-res", action="store_true",
                        help="Пропустить ресурсы")

    args = parser.parse_args()

    jadx_options = []
    if args.show_bad_code:
        jadx_options.append("--show-bad-code")
    if args.no_imports:
        jadx_options.append("--no-imports")
    if args.no_res:
        jadx_options.append("--no-res")

    decompiler = ApkDecompiler(jadx_path=args.jadx, output_dir=args.output)

    if args.file:
        decompiler.decompile_apk(args.file, jadx_options)
    elif args.directory:
        decompiler.batch_decompile(args.directory, jadx_options)


if __name__ == "__main__":
    main()
