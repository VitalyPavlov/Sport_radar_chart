import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.app
def test_streamlit_app_initial_state():
    """
    Test the initial state of the Streamlit app
    """

    # Create an AppTest instance
    at = AppTest.from_file("./app/app.py")
    at.run()

    assert not at.exception
