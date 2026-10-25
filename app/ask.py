"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（第1回 宿題）

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(client: genai.Client, history: list[types.Content], question: str) -> str:
    """history（これまでの会話）に question を足して送り、成功したら history を更新する。"""
    user_content = types.Content(role="user", parts=[types.Part(text=question)])
    # APIはステートレスなので、毎回これまでの会話を丸ごと送る
    response = client.models.generate_content(
        model=MODEL, contents=[*history, user_content]
    )

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] 入力トークン: {usage.prompt_token_count}, "
            f"出力トークン: {usage.candidates_token_count}"
        )

    # 成功したときだけ履歴に追加する（失敗時に user だけ残ると履歴が壊れるため）
    history.append(user_content)
    history.append(response.candidates[0].content)
    return response.text


def chat(client: genai.Client, first_question: str | None) -> None:
    """会話履歴を保ったまま、続けて質問を受け付ける。"""
    history: list[types.Content] = []
    print("会話モードです。空行か exit で終了します。")
    question = first_question
    if question is not None:
        print(f"\n[質問] {question}")
    while True:
        if question is None:
            try:
                question = input("\n[質問] ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if question in ("", "exit", "quit"):
            break
        try:
            print(ask(client, history, question))
        except errors.APIError as e:
            # 失敗しても history は変わっていないので、そのまま次の質問へ進める
            print(api_error_message(e.code, e.message))
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument(
        "--chat", action="store_true", help="会話履歴を保ったまま続けて質問する"
    )
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    print(f"[model] {MODEL}")
    if args.chat:
        chat(client, args.question)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(client, [], question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
