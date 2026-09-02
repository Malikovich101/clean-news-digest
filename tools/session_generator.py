from __future__ import annotations

from getpass import getpass

from telethon.sync import TelegramClient
from telethon.sessions import StringSession


api_id = int(input("Telegram API ID: ").strip())
api_hash = getpass("Telegram API Hash: ").strip()

print("\nОткроется авторизация Telegram. После входа скрипт выведет новую Telethon StringSession.")
print("Не сохраняй её в код и никому не отправляй.\n")

with TelegramClient(StringSession(), api_id, api_hash) as client:
    print("\n=== НОВАЯ TG_SESSION_STRING ===")
    print(client.session.save())
    print("=== КОНЕЦ TG_SESSION_STRING ===")
