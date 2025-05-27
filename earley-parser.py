from collections import deque, defaultdict
from itertools import count
from string import punctuation
from tabulate import tabulate


class Rule:
    """
    Represents a grammar rule in the form LHS -> RHS.
    The 'dot' indicates the current position in the RHS during parsing.
    """

    def __init__(self, lhs, rhs, dot):
        self.lhs = lhs
        self.rhs = rhs
        self.dot = dot

    def __str__(self):
        before_dot = " ".join(self.rhs[: self.dot])
        after_dot = " ".join(self.rhs[self.dot :])
        return f"{self.lhs} -> {before_dot} • {after_dot}".replace("  ", " ").strip()


class Edge:
    """Represents a single entry in the Earley parsing chart."""

    def __init__(self, id, rule, span, history):
        self.id = id
        self.rule = rule
        self.span: tuple[int, int] = span
        self.history = history

    def to_dict(self):
        return {
            "ID": self.id,
            "RULE": str(self.rule),
            "[start, end]": self.span,
            "HIST": ", ".join((str(x) for x in self.history)),
        }


class EarleyParser:
    def __init__(self, syntax, lexicon, sentence):
        self.syntax = defaultdict(
            list, {k: [tuple(s.split()) for s in v] for k, v in syntax.items()}
        )
        self.lexicon = defaultdict(list, lexicon)
        self.sentence = (
            sentence.translate(str.maketrans("", "", punctuation)).lower().split()
        )
        self.edge_id = count()  # next edge ID in table
        self.sentence_progress = 0  # how many words have been matched
        self.chart = [Edge(next(self.edge_id), Rule("S", ("NP", "VP"), 0), (0, 0), [])]
        """
        We start each iteration (n + 1) of the algorithm from 
        the start of the complete step of the iteration n 
        (where edges which have matched n words) 
        """
        self.complete_start = 0  # start of complete step for the previous word
        self.parse_count = 0  # how many successful parses

    def predict(self):
        """
        The predict step of the Earley algorithm.
        For any edge A -> α • B β, where B is a non-terminal,
        add new edges B -> • γ for all rules with B on the LHS.
        These new edges start where the current edge ends.
        """
        # edges pending prediction (completed edges from last stage)
        edge_queue = deque(self.chart[self.complete_start :])
        seen = set()

        while edge_queue:
            edge = edge_queue.popleft()
            # do not add completed edges
            if edge.rule.dot >= len(edge.rule.rhs):
                continue

            new_lhs = edge.rule.rhs[edge.rule.dot]

            if new_lhs in self.syntax:
                # if B is a non-terminal
                for new_rhs in self.syntax[new_lhs]:
                    if (new_lhs, new_rhs) not in seen:
                        # create predicted edge: B -> • γ
                        new_edge = Edge(
                            next(self.edge_id),
                            Rule(new_lhs, new_rhs, 0),
                            (edge.span[1], edge.span[1]),
                            [],
                        )
                        # add new_edge to predict_queue
                        edge_queue.append(new_edge)
                        self.chart.append(new_edge)
                        seen.add((new_lhs, new_rhs))
            else:
                # if B is a terminal (pre-scan)
                for new_word in self.lexicon[new_lhs]:
                    # scan ahead to avoid creating unnecessary edges
                    if new_word == self.sentence[self.sentence_progress]:
                        if (new_lhs, new_word) not in seen:
                            # create predicted edge B -> • new_word
                            new_edge = Edge(
                                next(self.edge_id),
                                Rule(new_lhs, (new_word,), 0),
                                (edge.span[1], edge.span[1]),
                                [],
                            )
                            self.chart.append(new_edge)
                            seen.add((new_lhs, new_word))

    def scan(self):
        """
        The scan step of the Earley algorithm.
        If an edge A -> α • T β is present, where T is a terminal (part-of-speech)
        that matches the current word in the input sentence,
        create a new edge A -> α T • β.
        This advances the dot over the terminal and extends the span.
        """
        scanned_edges = deque([])
        for edge in self.chart[self.complete_start :]:
            if edge.rule.lhs in self.lexicon:
                """
                move the • when you find a non-terminal in you lexicon.
                we know from the predict step that its expansion must
                match the sentence
                """
                edge.rule.dot += 1
                edge.span = (edge.span[0], edge.span[1] + 1)
                scanned_edges.append(edge)

        self.sentence_progress += 1
        self.complete_start = len(self.chart)
        return scanned_edges

    def complete(self, completed_edges):
        """
        The complete step of the Earley algorithm.
        If an edge B -> γ • is complete, find all pending edges A -> α • B β
        where the complete edge's start matches the pending edge's end.
        Create new edges A -> α B • β.
        """
        while completed_edges:
            completed_edge = completed_edges.popleft()
            for pending_edge in self.chart:
                """
                if pending_edge in incomplete and
                pending_edge ends where completed_edge begins and
                next non-terminal to process in pending_edge matches
                LHS of completed_edge
                """
                if (
                    pending_edge.rule.dot < len(pending_edge.rule.rhs)
                    and pending_edge.span[1] == completed_edge.span[0]
                    and pending_edge.rule.rhs[pending_edge.rule.dot]
                    == completed_edge.rule.lhs
                ):
                    # construct new edge by merging a pending and complete edge
                    new_edge = Edge(
                        next(self.edge_id),
                        Rule(
                            pending_edge.rule.lhs,
                            pending_edge.rule.rhs,
                            pending_edge.rule.dot + 1,
                        ),
                        (pending_edge.span[0], completed_edge.span[1]),
                        pending_edge.history + [completed_edge.id],
                    )
                    self.chart.append(new_edge)

                    # check if we have completed parsing the whole the sentence
                    if new_edge.rule.dot == len(new_edge.rule.rhs):
                        if new_edge.rule.lhs == "S" and self.sentence_progress == len(
                            self.sentence
                        ):
                            self.parse_count += 1
                        else:
                            completed_edges.append(new_edge)

    def run(self):
        """
        Runs the Earley parsing algorithm through its main loop:
        PREDICT -> SCAN -> COMPLETE for each word in the sentence.
        Finally, prints the chart and the number of parses.
        """
        while self.sentence_progress < len(self.sentence):
            self.predict()
            scanned_edges = self.scan()
            self.complete(scanned_edges)

        print(
            tabulate(
                [edge.to_dict() for edge in self.chart],
                headers="keys",
                tablefmt="plain",
                stralign="left",
                numalign="left",
            )
        )

        print(f"Parse Count: {self.parse_count}")


if __name__ == "__main__":
    """Example Usage"""

    syntax = {
        "S": ["NP VP"],
        "NP": ["N PP", "N"],
        "PP": ["P NP"],
        "VP": ["VP PP", "V VP", "V NP", "V"],
    }

    lexicon = {
        "N": ["they", "can", "fish", "rivers", "december"],
        "P": ["in"],
        "V": ["can", "fish"],
    }

    sentence1 = "They can fish in rivers."
    sentence2 = "They can fish in rivers in December."

    earley_parser = EarleyParser(syntax, lexicon, sentence2)
    earley_parser.run()
