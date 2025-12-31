import subprocess
import sys

# 运行目标脚本并捕获输出
try:
    result = subprocess.run(
        [sys.executable, r"c:\gemini-chat(time-model-regen)\test2_phase2.py"],
        capture_output=True,
        text=True,
        timeout=300  # 设置5分钟超时
    )
    
    print("=== 标准输出 ===")
    print(result.stdout)
    
    print("\n=== 标准错误 ===")
    print(result.stderr)
    
    print(f"\n=== 退出码 ===")
    print(result.returncode)
    
    # 保存到文件
    with open("test_output.log", "w") as f:
        f.write("=== 标准输出 ===\n")
        f.write(result.stdout)
        f.write("\n=== 标准错误 ===\n")
        f.write(result.stderr)
        f.write(f"\n=== 退出码 ===\n")
        f.write(str(result.returncode))
        
except subprocess.TimeoutExpired:
    print("脚本运行超时！")
except Exception as e:
    print(f"运行脚本时出错：{e}")