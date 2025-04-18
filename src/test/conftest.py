import pytest


def pytest_addoption(parser):
    parser.addoption("--code", action="store_true", default=False, help="run code tests")
    parser.addoption("--app",  action="store_true", default=False, help="run app tests" )

def pytest_collection_modifyitems(config, items):
    flags = {"code": "--code", "app": "--app"}
    skips = {keyword: pytest.mark.skip(reason="need {} option to run".format(flag))
             for keyword, flag in flags.items() if not config.getoption(flag)}
    
    for item in items:
        for keyword, skip in skips.items():
            if keyword in item.keywords:
                item.add_marker(skip)
