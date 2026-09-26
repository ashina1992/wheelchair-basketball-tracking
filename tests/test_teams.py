import numpy as np

from wbt.teams import assign_teams, rgb_to_lab

WHITE, RED, GREY = rgb_to_lab((235, 235, 235)), rgb_to_lab((200, 30, 40)), rgb_to_lab((150, 150, 150))


def obs(color, aspect, n=6, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return [(color + rng.normal(0, noise, 3), aspect) for _ in range(n)]


def test_assign_teams_with_profile_colors_and_standing_referee():
    track_obs = {1: obs(WHITE, 1.5, seed=1), 2: obs(WHITE, 1.4, seed=2), 3: obs(RED, 1.6, seed=3),
                 4: obs(RED, 1.3, seed=4), 9: obs(GREY, 2.6, seed=5)}
    got = assign_teams(track_obs, {"a": (235, 235, 235), "b": (200, 30, 40)})
    assert got == {1: "a", 2: "a", 3: "b", 4: "b", 9: "other"}


def test_assign_teams_without_colors_splits_two_clusters():
    track_obs = {1: obs(WHITE, 1.5, n=8, seed=1), 2: obs(RED, 1.5, seed=2), 3: obs(WHITE, 1.4, seed=3)}
    got = assign_teams(track_obs)
    assert got[1] == got[3] != got[2]


def test_short_tracks_are_ignored():
    track_obs = {1: obs(WHITE, 1.5, n=2), 2: obs(RED, 1.5, n=12), 3: obs(WHITE, 1.5, n=12)}
    assert 1 not in assign_teams(track_obs)
