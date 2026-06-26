import pytest
from pydantic import ValidationError
from domain.models import CardRegion, VoterCard, PageType

def test_card_region_validation():
    # Valid region
    region = CardRegion(x=10, y=20, w=100, h=50)
    assert region.x == 10
    assert region.w == 100

    # Invalid region (negative width)
    with pytest.raises(ValidationError):
        CardRegion(x=10, y=20, w=-5, h=50)

def test_voter_card_validation():
    # Valid card
    card = VoterCard(card_index=5, epic_id="ABC1234567", name="John Doe")
    assert card.card_index == 5
    assert card.epic_id == "ABC1234567"
    assert card.name == "John Doe"

    # Invalid card_index (must be >= 1)
    with pytest.raises(ValidationError):
        VoterCard(card_index=0)

    # Invalid card_index (must be <= 30)
    with pytest.raises(ValidationError):
        VoterCard(card_index=31)

def test_page_type_enum():
    assert PageType.METADATA == "METADATA"
    assert PageType.VOTER_LIST == "VOTER_LIST"
    assert PageType.SUMMARY == "SUMMARY"
    assert PageType.BLANK == "BLANK"
