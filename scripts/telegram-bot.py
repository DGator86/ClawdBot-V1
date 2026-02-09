#!/usr/bin/env python3
"""
Telegram Bot — Single-Command Entry Point.
============================================
Runs the full ClawdBot × Yoshi × Kalshi × Ralph system
with Telegram push notifications and command interface.

    One command:
        python3 scripts/telegram-bot.py

    What it does:
        - Receives commands: /scan /status /ralph /params /help
        - Runs the orchestrator in the background (every 60s)
        - Pushes BUY alerts to your phone automatically
        - Ralph learns and optimizes continuously

    Setup:
        1. Create a bot via @BotFather on Telegram
        2. Add to .env:
             TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
        3. Run this script
        4. Send any message to your bot — it will auto-discover your chat ID
        5. Add TELEGRAM_CHAT_ID=your_chat_id to .env

    Options:
        --interval N     Scan interval in seconds (default: 60)
        --no-scanner     Bot only, no background scanning
        --no-kalshi      Skip Kalshi market scanning
        --no-forecast    Skip ClawdBot forecast
        --notify-all     Send full cycle reports (not just BUY alerts)
        --token TOKEN    Bot token (override .env)
        --chat-id ID     Chat ID (override .env)
"""
import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(
        description="ClawdBot Telegram Bot — Trading alerts & commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--interval", type=float, default=60,
                    help="Scan interval in seconds (default: 60)")
    p.add_argument("--no-scanner", action="store_true",
                    help="Run bot without background scanner")
    p.add_argument("--no-kalshi", action="store_true",
                    help="Skip Kalshi scanning")
    p.add_argument("--no-forecast", action="store_true",
                    help="Skip ClawdBot forecast")
    p.add_argument("--notify-all", action="store_true",
                    help="Send full cycle reports (not just BUY alerts)")
    p.add_argument("--token", type=str, default="",
                    help="Bot token (override .env)")
    p.add_argument("--chat-id", type=str, default="",
                    help="Chat ID (override .env)")
    p.add_argument("--explore", type=float, default=0.10,
                    help="Ralph exploration rate (default: 0.10)")
    p.add_argument("--series", type=str, default="KXBTC,KXETH",
                    help="Kalshi series (default: KXBTC,KXETH)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load .env
    for env_path in [
        os.path.join(os.getcwd(), ".env"),
        "/root/ClawdBot-V1/.env",
        os.path.expanduser("~/.env"),
    ]:
        if os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and val:
                            os.environ.setdefault(key, val)
            except Exception:
                pass

    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = args.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token:
        print("=" * 55)
        print("  TELEGRAM BOT — Setup Required")
        print("=" * 55)
        print()
        print("  No TELEGRAM_BOT_TOKEN found.")
        print()
        print("  Quick setup:")
        print("    1. Open Telegram, search for @BotFather")
        print("    2. Send /newbot and follow prompts")
        print("    3. Copy the token (looks like 123456:ABCdef...)")
        print("    4. Add to .env:")
        print("         TELEGRAM_BOT_TOKEN=your_token_here")
        print("    5. Re-run this script")
        print()
        print("  Or pass directly:")
        print("    python3 scripts/telegram-bot.py --token YOUR_TOKEN")
        print()
        print("=" * 55)
        sys.exit(1)

    # Build orchestrator (unless --no-scanner)
    orchestrator = None
    if not args.no_scanner:
        from gnosis.orchestrator import UnifiedOrchestrator, OrchestratorConfig
        from gnosis.ralph.learner import LearningConfig

        series = [s.strip() for s in args.series.split(",") if s.strip()]

        config = OrchestratorConfig(
            kalshi_series=series,
            horizon_hours=1.0,
            learning=LearningConfig(
                explore_rate=args.explore,
                verbose=False,  # quiet — bot handles output
            ),
            cycle_interval_s=args.interval,
            enable_forecast=not args.no_forecast,
            enable_kalshi=not args.no_kalshi,
            enable_ralph=True,
            verbose=False,
        )
        orchestrator = UnifiedOrchestrator(config=config)

    # Create and run bot
    from gnosis.telegram.bot import TelegramBot

    bot = TelegramBot(
        token=token,
        chat_id=chat_id,
        orchestrator=orchestrator,
        auto_scan_interval=args.interval,
        notify_on_buy=True,
        notify_on_cycle=args.notify_all,
    )

    print("=" * 55)
    print("  CLAWDBOT TELEGRAM BOT")
    print("=" * 55)
    print(f"  Scanner:  {'ON' if orchestrator else 'OFF'}")
    print(f"  Interval: {args.interval}s")
    print(f"  Kalshi:   {'ON' if not args.no_kalshi else 'OFF'}")
    print(f"  Forecast: {'ON' if not args.no_forecast else 'OFF'}")
    print(f"  Alerts:   BUY signals{' + full reports' if args.notify_all else ''}")
    if chat_id:
        print(f"  Chat ID:  {chat_id}")
    else:
        print(f"  Chat ID:  (auto-discover — send a message to the bot)")
    print("=" * 55)

    bot.run()


if __name__ == "__main__":
    main()
