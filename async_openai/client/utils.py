from typing import Dict, Union, Optional

def parse_stream(body: Union[str, bytes]):
    for line in body:
        if line:
            if line == b"data: [DONE]":
                # return here will cause GeneratorExit exception in urllib3
                # and it will close http connection with TCP Reset
                continue
            if hasattr(line, "decode"):
                line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[len("data: ") :]
            yield line


def build_proxies(
    proxy: Optional[Union[str, Dict]] = None,
):
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http://": proxy, "https://": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise TypeError("Proxy must be a string or a dictionary")


