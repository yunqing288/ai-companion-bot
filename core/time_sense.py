"""时间感知（共享底层）：被动响应与主动自发两大系统共用的“现在几点 / 什么时段 / 隔了多久”。

把原先散落在 telegram_bot.py 的四处时间逻辑统一收口于此：
  1. 时区检测与持有（_detect_timezone / TIMEZONE）
  2. 被动：注入给主模型的当前时间块（current_time_block）
  3. 主动：睡眠时段判断（is_sleeping / is_deep_sleep）、整点对齐（minutes_until_next_hour）
  4. 距上次发言多久的人类可读串（human_gap）

红线：本模块只做“读时间 → 给判断 / 文本”，不碰记忆、召回、人设。
“现在几点”的标准表述只在 format_clock 一处定义，被动与主动共用，避免格式漂移。
"""
import sys
import time as _t
import subprocess as _sp
from datetime import datetime
from zoneinfo import ZoneInfo

# ── 睡眠时段参数（主动系统用）──────────────────────────────
SLEEP_START_HOUR = 1   # 含：[SLEEP_START, SLEEP_END) 视为睡着
SLEEP_END_HOUR = 9     # 不含
DEEP_SLEEP_BEFORE = 6  # 小时 < 此值算深睡，否则浅睡/将醒


def _detect_timezone() -> ZoneInfo:
    """跨平台自动探测系统时区，失败逐级兜底，最终回落 UTC。"""
    # 方法1: macOS/Linux — 从 /etc/localtime 符号链接读取
    if sys.platform != "win32":
        try:
            tz_name = _sp.check_output(
                ["readlink", "/etc/localtime"], text=True, stderr=_sp.DEVNULL
            ).strip().split("zoneinfo/")[-1]
            return ZoneInfo(tz_name)
        except Exception:
            pass
    # 方法2: Windows — tzutil 读时区名并映射到 IANA
    if sys.platform == "win32":
        try:
            tz_win = _sp.check_output(["tzutil", "/g"], text=True, stderr=_sp.DEVNULL).strip()
            win_to_iana = {
                "China Standard Time": "Asia/Shanghai",
                "Eastern Standard Time": "America/New_York",
                "Pacific Standard Time": "America/Los_Angeles",
                "Central Standard Time": "America/Chicago",
                "Mountain Standard Time": "America/Denver",
                "GMT Standard Time": "Europe/London",
                "W. Europe Standard Time": "Europe/Berlin",
                "Tokyo Standard Time": "Asia/Tokyo",
                "Korea Standard Time": "Asia/Seoul",
                "Taipei Standard Time": "Asia/Taipei",
                "Singapore Standard Time": "Asia/Singapore",
                "AUS Eastern Standard Time": "Australia/Sydney",
            }
            if tz_win in win_to_iana:
                return ZoneInfo(win_to_iana[tz_win])
        except Exception:
            pass
    # 方法3: tzlocal 库（装了才用）
    try:
        from tzlocal import get_localzone_name
        return ZoneInfo(get_localzone_name())
    except Exception:
        pass
    # 方法4: 用 UTC offset 算一个近似时区
    offset_hours = -(_t.timezone if _t.daylight == 0 else _t.altzone) // 3600
    common_offsets = {
        8: "Asia/Shanghai", 9: "Asia/Tokyo", -5: "America/New_York",
        -8: "America/Los_Angeles", -6: "America/Chicago", 0: "Europe/London",
        1: "Europe/Berlin", -4: "America/New_York",
    }
    tz_name = common_offsets.get(offset_hours, "UTC")
    print(f"[时区] 自动检测: UTC{'+' if offset_hours >= 0 else ''}{offset_hours} → {tz_name}")
    return ZoneInfo(tz_name)


# 进程内持有的时区（import 时定一次，全程复用）
TIMEZONE = _detect_timezone()


def now() -> datetime:
    """带时区的当前时间（统一入口）。"""
    return datetime.now(TIMEZONE)


def format_clock(dt: datetime = None) -> str:
    """“现在几点”的标准表述，被动注入与主动 prompt 共用一处定义。"""
    return (dt or now()).strftime("%Y年%m月%d日 %H:%M")


def current_time_block() -> str:
    """被动：拼给主模型的当前时间 system 块（含防猜测提示）。顺手打印注入日志。"""
    n = now()
    print(f"[时间注入] {n.strftime('%Y-%m-%d %H:%M %Z')}")
    return (
        f"<current_time>{format_clock(n)}</current_time>\n"
        f"当被问到时间或日期时，直接告知上方 current_time 里的准确时间，不要猜测。"
    )


def is_sleeping(dt: datetime = None) -> bool:
    """主动：当前是否处于睡眠时段 [SLEEP_START, SLEEP_END)。"""
    return SLEEP_START_HOUR <= (dt or now()).hour < SLEEP_END_HOUR


def is_deep_sleep(dt: datetime = None) -> bool:
    """主动：睡眠时段内是否深睡（用于挑“睡着”还是“将醒”类活动）。"""
    return (dt or now()).hour < DEEP_SLEEP_BEFORE


def human_gap(since_iso: str, ref: datetime = None) -> str:
    """距 since_iso（ISO 时间字符串）多久的人类可读串：‘X分钟前’/‘X.X小时前’/‘未知’。

    兼容裸时间戳（旧数据无时区）：相减抛错时返回‘未知’，不让调用方崩。
    """
    if not since_iso:
        return "未知"
    try:
        gap = (ref or now()) - datetime.fromisoformat(since_iso)
        hours = gap.total_seconds() / 3600
        if hours < 1:
            return f"{int(gap.total_seconds() / 60)}分钟前"
        return f"{hours:.1f}小时前"
    except Exception:
        return "未知"


def minutes_until_next_hour(dt: datetime = None) -> int:
    """主动：距下一个整点还有多少分钟（life tick 对齐整点用），整点时返回 0。"""
    m = 60 - (dt or now()).minute
    return 0 if m == 60 else m
