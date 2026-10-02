"""The panel answers only to its own names, plus those a proxy passes on."""

import pytest

from lllm2 import config
from lllm2.app import host_allowed, origin_allowed

OWN = {"127.0.0.1:8082", "localhost:8082"}
LISTED = frozenset({"lllm2.example.com"})


@pytest.mark.parametrize("environ", [{}, {"LLLM2_PANEL_ALLOWED_HOSTS": " , "}])
def test_no_extra_names_by_default(environ):
    assert config.panel_allowed_hosts(environ) == frozenset()


def test_extra_names_are_trimmed_and_lower_cased():
    environ = {"LLLM2_PANEL_ALLOWED_HOSTS": " A.example.com, b.example.com "}
    assert config.panel_allowed_hosts(environ) == {"a.example.com", "b.example.com"}


@pytest.mark.parametrize(
    "value", ["https://x.example.com", "x.example.com:443", "*.example.com", "a b"]
)
def test_an_extra_name_must_be_a_bare_host_name(value):
    with pytest.raises(ValueError, match="LLLM2_PANEL_ALLOWED_HOSTS"):
        config.panel_allowed_hosts({"LLLM2_PANEL_ALLOWED_HOSTS": value})


@pytest.mark.parametrize(
    "header",
    [
        "localhost:8082",
        "LOCALHOST:8082",
        "lllm2.example.com",
        "LLLM2.Example.com",
        # A controller on a NodePort passes the port on.
        "lllm2.example.com:30443",
    ],
)
def test_the_panel_answers_to_its_own_and_listed_names(header):
    assert host_allowed(header, OWN, LISTED)


@pytest.mark.parametrize(
    "header",
    [
        "evil.example.com",
        "lllm2.example.com.evil.com",
        "evil.com:8082",
        # Its own names need its own port, as before.
        "localhost",
        "localhost:9999",
        "lllm2.example.com:",
        "lllm2.example.com:https",
        "",
    ],
)
def test_the_panel_refuses_other_names(header):
    assert not host_allowed(header, OWN, LISTED)


def test_a_listed_name_needs_the_variable():
    assert not host_allowed("lllm2.example.com", OWN)


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        (None, "lllm2.example.com"),
        ("", "lllm2.example.com"),
        ("http://127.0.0.1:8082", "127.0.0.1:8082"),
        # A browser behind a TLS Ingress sends https with the same host.
        ("https://lllm2.example.com", "lllm2.example.com"),
        ("HTTPS://LLLM2.example.com", "lllm2.example.com"),
        ("https://lllm2.example.com:30443", "lllm2.example.com:30443"),
    ],
)
def test_a_post_from_the_panels_own_page_is_accepted(origin, host):
    assert origin_allowed(origin, host)


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        ("https://evil.example.com", "lllm2.example.com"),
        ("https://lllm2.example.com.evil.com", "lllm2.example.com"),
        ("https://lllm2.example.com", "lllm2.example.com:30443"),
        ("null", "lllm2.example.com"),
    ],
)
def test_a_post_from_another_page_is_refused(origin, host):
    assert not origin_allowed(origin, host)
