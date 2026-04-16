from fit.advisor import Advisor


def test_advisor_is_abstract():
    """Advisor cannot be instantiated directly."""
    try:
        Advisor()
        assert False, "should raise"
    except TypeError:
        pass
