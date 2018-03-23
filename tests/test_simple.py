from pytopicrank import TopicRank


def test_example():
    with open('tests/ion_exchange.txt') as f:
        text = f.read()
        tr = TopicRank(text)
        assert tr.get_top_n(n=2) == ['exchang ion', 'mathemat model']
