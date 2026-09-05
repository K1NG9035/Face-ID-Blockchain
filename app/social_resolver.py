from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from urllib.parse import urlparse
from typing import Any


@dataclass(frozen=True)
class SocialPostMetadata:
    platform: str
    is_social_post: bool
    author: str | None
    post_url: str | None
    image_url: str
    caption: str | None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SOCIAL_DOMAINS = {
    "x.com": "Twitter/X",
    "twitter.com": "Twitter/X",
    "twimg.com": "Twitter/X",
    "instagram.com": "Instagram",
    "cdninstagram.com": "Instagram",
    "reddit.com": "Reddit",
    "redd.it": "Reddit",
    "linkedin.com": "LinkedIn",
    "licdn.com": "LinkedIn",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "ytimg.com": "YouTube",
    "pinterest.com": "Pinterest",
    "pinimg.com": "Pinterest",
    "facebook.com": "Facebook",
    "fbcdn.net": "Facebook",
    "tiktok.com": "TikTok",
}


def detect_platform(url: str) -> tuple[str, bool]:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    for domain, platform in _SOCIAL_DOMAINS.items():
        if hostname == domain or hostname.endswith("." + domain):
            return platform, True
    return "Web", False


def resolve_social_post(candidate_url: str, page_url: str | None = None) -> SocialPostMetadata:
    """Analyze candidate and page URLs to reconstruct social media post context."""
    target_url = page_url or candidate_url
    platform, is_social = detect_platform(target_url)
    if not is_social and page_url:
        platform, is_social = detect_platform(page_url)
        if is_social:
            target_url = page_url

    author: str | None = None
    caption: str | None = None
    post_url: str | None = page_url if page_url and page_url.startswith("http") else None

    parsed = urlparse(target_url)
    path = parsed.path.strip("/")
    parts = path.split("/") if path else []

    if platform == "Twitter/X":
        if len(parts) >= 1 and parts[0] not in {"i", "intent", "search", "hashtag", "explore"}:
            author = f"@{parts[0]}"
        if len(parts) >= 3 and parts[1] == "status":
            post_url = f"https://x.com/{parts[0]}/status/{parts[2]}"
            caption = f"Post by {author} on Twitter/X"

    elif platform == "Reddit":
        if len(parts) >= 2 and parts[0] == "r":
            author = f"r/{parts[1]}"
            if len(parts) >= 5 and parts[2] == "comments":
                title_slug = parts[4].replace("_", " ").title()
                caption = title_slug[:80]
                post_url = f"https://reddit.com/r/{parts[1]}/comments/{parts[3]}/{parts[4]}"
        elif len(parts) >= 2 and parts[0] in {"u", "user"}:
            author = f"u/{parts[1]}"

    elif platform == "Instagram":
        if len(parts) >= 2 and parts[0] == "p":
            post_url = f"https://www.instagram.com/p/{parts[1]}/"
            caption = "Instagram photo post"
        elif len(parts) >= 1 and parts[0] not in {"explore", "reels", "stories"}:
            author = f"@{parts[0]}"

    elif platform == "LinkedIn":
        if len(parts) >= 2 and parts[0] == "in":
            author = parts[1]
            post_url = f"https://www.linkedin.com/in/{parts[1]}/"
        elif len(parts) >= 2 and parts[0] == "posts":
            post_url = f"https://www.linkedin.com/posts/{parts[1]}"

    elif platform == "YouTube":
        if len(parts) >= 1 and parts[0].startswith("@"):
            author = parts[0]
        elif "v=" in parsed.query:
            post_url = target_url
            caption = "YouTube Video frame"

    elif platform == "Pinterest":
        if len(parts) >= 2 and parts[0] == "pin":
            post_url = f"https://www.pinterest.com/pin/{parts[1]}/"
            caption = "Pinterest Pin"

    if not post_url and is_social and target_url.startswith("http"):
        post_url = target_url

    if not caption and post_url:
        caption = f"Content found on {platform}"

    return SocialPostMetadata(
        platform=platform,
        is_social_post=is_social,
        author=author,
        post_url=post_url,
        image_url=candidate_url,
        caption=caption,
    )
