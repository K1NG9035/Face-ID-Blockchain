import pytest

from app.social_resolver import detect_platform, resolve_social_post


def test_detect_platform_social_domains():
    assert detect_platform("https://x.com/elonmusk/status/123456")[0] == "Twitter/X"
    assert detect_platform("https://twitter.com/NASA")[0] == "Twitter/X"
    assert detect_platform("https://reddit.com/r/technology/comments/abc/news")[0] == "Reddit"
    assert detect_platform("https://www.instagram.com/p/C12345/")[0] == "Instagram"
    assert detect_platform("https://www.linkedin.com/in/satyanadella/")[0] == "LinkedIn"
    assert detect_platform("https://www.youtube.com/watch?v=dQw4w9WgXcQ")[0] == "YouTube"
    assert detect_platform("https://example.com/article.html")[0] == "Web"


def test_resolve_social_post_twitter():
    meta = resolve_social_post(
        candidate_url="https://pbs.twimg.com/media/F123.jpg",
        page_url="https://x.com/JohnDoe/status/987654321",
    )
    assert meta.platform == "Twitter/X"
    assert meta.is_social_post is True
    assert meta.author == "@JohnDoe"
    assert meta.post_url == "https://x.com/JohnDoe/status/987654321"
    assert "JohnDoe" in (meta.caption or "")


def test_resolve_social_post_reddit():
    meta = resolve_social_post(
        candidate_url="https://i.redd.it/photo.jpg",
        page_url="https://reddit.com/r/pics/comments/xyz123/sunset_in_goa",
    )
    assert meta.platform == "Reddit"
    assert meta.author == "r/pics"
    assert "Sunset In Goa" in (meta.caption or "")


def test_resolve_social_post_generic_web():
    meta = resolve_social_post(
        candidate_url="https://images.example.com/photo.jpg",
        page_url="https://news.example.com/article/1",
    )
    assert meta.platform == "Web"
    assert meta.is_social_post is False
    assert meta.author is None
    assert meta.post_url == "https://news.example.com/article/1"
