import os
from datetime import datetime


def format_log_message(message: str) -> str:
    """Format pesan log dengan timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] {message}"


def print_log(message: str, print_tag: bool = True) -> str:
    """Print log ke console dan kembalikan string yang sudah diformat"""
    log_str = format_log_message(message)
    
    if print_tag:
        print(f"\n{log_str}")
    
    return log_str


def write_log_to_file(output_path: str, filename: str, log_content: str, append: bool = True):
    """
    Tulis log ke file.
    - append=True  → tambahkan ke file yang sudah ada
    - append=False → buat file baru (overwrite)
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, f"{filename}.txt")
    mode = 'a' if append else 'w'
    
    with open(log_file, mode, encoding='utf-8') as f:
        f.write(log_content)


def append_log(output_path: str, filename: str, message: str, print_tag: bool = True):
    """
    Fungsi utama: simpan log sekaligus (print + tulis ke file)
    """
    log_str = print_log(message, print_tag)
    
    content_to_write = f"\n{log_str}"
    
    write_log_to_file(output_path, filename, content_to_write, append=True)