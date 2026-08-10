from datetime import datetime, timedelta, timezone

import pytest

from pmbot.config import DevigSettings
from pmbot.fairline import best_price, book_fair_probs, fair_prob_at
from pmbot.types import BookLine, PropMarket


def market(books, player="Test Player", sport="nba", market_key="player_points"):
    return PropMarket(
        sport=sport,
        event_id="evt-1",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=3),
        home_team="HOME",
        away_team="AWAY",
        player=player,
        market=market_key,
        books=tuple(books),
        team="AWAY",
        opponent="HOME",
    )


def line(book, ln, over, under):
    return BookLine(book=book, line=ln, over_american=over, under_american=under)


class TestBookFairProbs:
    def test_each_book_is_devigged_separately(self):
        m = market([line("pinnacle", 25.5, -110, -110), line("fanduel", 25.5, -130, 105)])
        fair = book_fair_probs(m, DevigSettings())
        assert len(fair) == 2
        assert fair[0].p_over == pytest.approx(0.5)
        assert fair[1].p_over > 0.5
        assert all(0 < f.hold < 0.1 for f in fair)

    def test_one_sided_quotes_are_skipped(self):
        m = market([line("pinnacle", 25.5, -110, None)])
        assert book_fair_probs(m, DevigSettings()) == []

    def test_sharp_books_carry_more_weight(self):
        m = market([line("pinnacle", 25.5, -110, -110), line("betmgm", 25.5, -110, -110)])
        weights = {f.book: f.weight for f in book_fair_probs(m, DevigSettings())}
        assert weights["pinnacle"] > weights["betmgm"]


class TestFairProbAt:
    def test_consensus_of_books_on_the_same_line(self):
        m = market([
            line("pinnacle", 25.5, -110, -110),
            line("fanduel", 25.5, -110, -110),
            line("betmgm", 25.5, -110, -110),
        ])
        fair = fair_prob_at(m, 25.5, DevigSettings())
        assert fair is not None
        assert fair.p_over == pytest.approx(0.5, abs=1e-6)
        assert fair.n_books == 3
        assert not fair.interpolated

    def test_the_book_being_priced_is_excluded(self):
        # Four books at -110 and one hanging +140. Including the outlier in
        # its own fair line would hide most of the edge.
        books = [line(b, 25.5, -110, -110) for b in ("pinnacle", "fanduel", "betmgm", "caesars")]
        books.append(line("draftkings", 25.5, 140, -175))
        m = market(books)

        excluded = fair_prob_at(m, 25.5, DevigSettings(), exclude_book="draftkings")
        included = fair_prob_at(m, 25.5, DevigSettings(exclude_own_book=False), exclude_book="draftkings")
        assert excluded.p_over == pytest.approx(0.5, abs=1e-6)
        assert included.p_over < excluded.p_over
        assert "draftkings" not in excluded.books_used

    def test_exclusion_falls_back_when_it_would_empty_the_market(self):
        m = market([line("draftkings", 25.5, 140, -175)])
        fair = fair_prob_at(m, 25.5, DevigSettings(), exclude_book="draftkings")
        assert fair is not None
        assert any("includes draftkings" in w for w in fair.warnings)

    def test_interpolates_between_different_lines(self):
        m = market([
            line("pinnacle", 25.5, -140, 120),
            line("fanduel", 27.5, 120, -140),
        ])
        fair = fair_prob_at(m, 26.5, DevigSettings())
        assert fair is not None
        assert fair.interpolated
        assert fair.p_over == pytest.approx(0.5, abs=0.02)
        assert fair.slope < 0  # a higher line is always a lower over probability

    def test_probability_falls_as_the_line_rises(self):
        m = market([
            line("pinnacle", 24.5, -160, 135),
            line("fanduel", 26.5, 110, -130),
        ])
        probs = [fair_prob_at(m, x, DevigSettings()).p_over for x in (24.5, 25.5, 26.5)]
        assert probs[0] > probs[1] > probs[2]

    def test_distant_lines_are_dropped(self):
        m = market([
            line("pinnacle", 25.5, -110, -110),
            line("fanduel", 40.5, -110, -110),  # different player-shaped universe
        ])
        fair = fair_prob_at(m, 25.5, DevigSettings(max_line_gap=2.5))
        assert fair is not None
        assert fair.n_books == 1
        assert any("too far" in w for w in fair.warnings)

    def test_returns_none_when_nothing_is_close_enough(self):
        m = market([line("pinnacle", 40.5, -110, -110)])
        assert fair_prob_at(m, 25.5, DevigSettings()) is None

    def test_single_line_cannot_be_extrapolated_elsewhere(self):
        m = market([line("pinnacle", 25.5, -110, -110), line("fanduel", 25.5, -105, -115)])
        assert fair_prob_at(m, 26.5, DevigSettings()) is None

    def test_non_monotone_books_fall_back_to_the_exact_line(self):
        # Books quoting a higher line with a *higher* over price is noise.
        m = market([
            line("pinnacle", 25.5, 150, -180),
            line("fanduel", 27.5, -180, 150),
            line("betmgm", 25.5, 145, -175),
        ])
        fair = fair_prob_at(m, 25.5, DevigSettings())
        assert fair is not None
        assert any("non-monotone" in w for w in fair.warnings)
        assert fair.p_over < 0.5

    def test_empty_market_gives_nothing(self):
        assert fair_prob_at(market([]), 25.5, DevigSettings()) is None


class TestBestPrice:
    def test_picks_the_longest_price_for_the_side(self):
        m = market([
            line("pinnacle", 25.5, -110, -110),
            line("caesars", 25.5, 140, -175),
            line("fanduel", 25.5, -105, -115),
        ])
        assert best_price(m, "over", 25.5) == ("caesars", 140)
        assert best_price(m, "under", 25.5) == ("pinnacle", -110)

    def test_only_considers_the_exact_line(self):
        m = market([line("caesars", 26.5, 200, -260), line("pinnacle", 25.5, -110, -110)])
        assert best_price(m, "over", 25.5) == ("pinnacle", -110)

    def test_missing_line_gives_nothing(self):
        m = market([line("pinnacle", 25.5, -110, -110)])
        assert best_price(m, "over", 30.5) is None
