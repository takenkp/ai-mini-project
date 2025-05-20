import os

# prompts 폴더에서 프롬프트를 로드하는 함수
def load_prompt_from_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        agent_name = os.path.splitext(os.path.basename(__file__))[0] 
        # 호출하는 __init__에서 FileNotFoundError를 발생시키므로, 여기서는 빈 문자열 반환 후 확인
        print(f"경고({agent_name}): 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
        return "" 
    except Exception as e:
        agent_name = os.path.splitext(os.path.basename(__file__))[0]
        print(f"오류({agent_name}): 프롬프트 파일 로드 중 문제 발생 - {file_path}: {e}")
        return ""
