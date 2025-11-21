import os
import subprocess
import sys

# 你的签到脚本的完整路径
script_path = r"C:\Users\dyd\PycharmProjects\PythonProject\sign_in.py"

# 检查是否安装 pyinstaller
try:
    import PyInstaller
except ImportError:
    print("未检测到 PyInstaller，正在自动安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# 构建 pyinstaller 命令
cmd = [
    "pyinstaller",
    "--noconsole",        # 不显示黑色终端窗口
    "--onefile",          # 打包成单个 exe 文件
    script_path
]

# 执行打包
print("开始打包...")
subprocess.run(cmd)

print("\n✅ 打包完成，请到以下目录查看 exe 文件：")
print("   dist\\sign_in.exe")
