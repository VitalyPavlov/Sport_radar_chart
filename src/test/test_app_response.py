import pytest
import requests

@pytest.mark.app
def test_streamlit_app_running():
    """
    Test that the Streamlit app is running and accessible
    """
    try:
        # Try to access the Streamlit app
        response = requests.get("http://localhost:8501")
        assert response.status_code == 200, "Streamlit app is not running"
    except requests.ConnectionError:
        pytest.fail("Could not connect to Streamlit app")
