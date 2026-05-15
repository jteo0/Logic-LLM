"""
SWI-Prolog replacement for the PyKE expert system.

Replaces: symbolic_solvers/pyke_solver/pyke_solver.py
Datasets: ProntoQA, ProofWriter

Approach:
  Like PyKE, each problem is written to a temp .pl file and consulted by
  SWI-Prolog. This properly executes :- table directives at load time,
  which enables tabling (memoization) to detect and terminate on circular
  rules — the key termination guarantee PyKE provided via forward chaining.

  Each instance gets its own module, preventing cross-contamination between
  problems in the same process.

DSL format (unchanged from original Logic-LM):
    Facts:  Predicate(Entity, True/False)
    Rules:  Predicate($x, True) && ... >>> Predicate($x, True)
    Query:  Predicate(Entity, True/False)

Encoding:
    Furry(Anne, True)  ->  furry_true(anne)
    Shy(Alex, False)   ->  shy_false(alex)
    Furry($x, True)    ->  furry_true(X)      (Prolog variable)
"""

import os
import re
import uuid
import tempfile
from pyswip import Prolog


class Prolog_Program:
    _engine = Prolog()  # one shared SWI-Prolog process

    def __init__(self, logic_program, dataset_name='ProntoQA'):
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.module = "m_" + uuid.uuid4().hex
        self.pl_file = None  # path to temp .pl file

        self.flag = self._parse_and_load()

        self.answer_map = {
            'ProntoQA': self._answer_map_prontoqa,
            'ProofWriter': self._answer_map_proofwriter,
        }

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_and_load(self):
        keywords = ['Query:', 'Rules:', 'Facts:', 'Predicates:']
        program_str = self.logic_program
        for keyword in keywords:
            try:
                program_str, segment = program_str.split(keyword)
                lines = [
                    l.split(':::')[0].strip()
                    for l in segment.strip().split('\n')
                ]
                setattr(self, keyword[:-1], lines)
            except Exception:
                setattr(self, keyword[:-1], None)

        if self.Facts is None or self.Rules is None or self.Query is None:
            return False

        try:
            pl_source = self._build_pl_source()
            self._consult(pl_source)
            return True
        except Exception as e:
            print(f"[Prolog_Program] Load error: {e}")
            return False

    # ------------------------------------------------------------------ #
    # DSL -> Prolog translation                                            #
    # ------------------------------------------------------------------ #

    def _dsl_atom_to_prolog(self, atom):
        """
        One-arg:  Furry(Anne, True)     ->  furry_true(anne)
                  Furry($x, True)       ->  furry_true(X)
        Two-arg:  Likes(Bear, Cat, True) -> likes_true(bear, cat)
                  Likes($x, $y, True)   -> likes_true(X, Y)
        """
        atom = atom.strip()
        # Try two-argument form first: Pred(arg1, arg2, True/False)
        m2 = re.match(r'(\w+)\(([^,]+),\s*([^,]+),\s*(True|False)\)', atom)
        if m2:
            predicate = m2.group(1).lower()
            arg1 = m2.group(2).strip()
            arg2 = m2.group(3).strip()
            value = m2.group(4)
            def to_prolog_arg(a):
                return a[1:].upper() if a.startswith('$') else a.lower()
            return f"{predicate}_{value.lower()}({to_prolog_arg(arg1)}, {to_prolog_arg(arg2)})"
        # Fall back to one-argument form: Pred(arg, True/False)
        m1 = re.match(r'(\w+)\(([^,]+),\s*(True|False)\)', atom)
        if m1:
            predicate = m1.group(1).lower()
            arg = m1.group(2).strip()
            value = m1.group(3)
            prolog_arg = arg[1:].upper() if arg.startswith('$') else arg.lower()
            return f"{predicate}_{value.lower()}({prolog_arg})"
        return None

    def _get_functor_and_arity(self, term):
        """Extract (functor_name, arity) from a Prolog term string."""
        if term is None:
            return None
        name = term.split('(')[0]
        arity = term.count(',') + 1  # number of commas + 1 = arity
        return (name, arity)

    def _collect_functors(self):
        """Collect all unique (functor, arity) pairs for tabling declarations."""
        functors = {}  # name -> arity
        for rule in (self.Rules or []):
            if '>>>' not in rule:
                continue
            _, conclusion_str = rule.split('>>>')
            for c in conclusion_str.split('&&'):
                term = self._dsl_atom_to_prolog(c.strip())
                fa = self._get_functor_and_arity(term)
                if fa:
                    functors[fa[0]] = fa[1]
        for fact in (self.Facts or []):
            term = self._dsl_atom_to_prolog(fact.strip())
            fa = self._get_functor_and_arity(term)
            if fa:
                functors[fa[0]] = fa[1]
        return functors

    def _build_pl_source(self):
        """Build the full Prolog source as a string."""
        lines = []

        # Module declaration
        lines.append(f':- module({self.module}, []).')
        lines.append('')

        # Table declarations for all derived predicates (prevents infinite loops)
        functors = self._collect_functors()  # {name: arity}
        for f in sorted(functors):
            arity = functors[f]
            lines.append(f':- table {f}/{arity}.')
        if functors:
            discontig = ', '.join(f'{f}/{functors[f]}' for f in sorted(functors))
            lines.append(f':- discontiguous {discontig}.')
        lines.append('')

        # Facts
        for fact in (self.Facts or []):
            fact = fact.strip()
            if not fact or '$' in fact:
                continue
            term = self._dsl_atom_to_prolog(fact)
            if term:
                lines.append(f'{term}.')

        lines.append('')

        # Rules
        for rule in (self.Rules or []):
            rule = rule.strip()
            if not rule or '>>>' not in rule:
                continue
            premise_str, conclusion_str = rule.split('>>>')
            premises = [p.strip() for p in premise_str.split('&&')]
            conclusions = [c.strip() for c in conclusion_str.split('&&')]
            prolog_premises = [self._dsl_atom_to_prolog(p) for p in premises]
            prolog_conclusions = [self._dsl_atom_to_prolog(c) for c in conclusions]
            if None in prolog_premises or None in prolog_conclusions:
                continue
            body = ', '.join(prolog_premises)
            for head in prolog_conclusions:
                lines.append(f'{head} :- {body}.')

        return '\n'.join(lines)

    def _consult(self, pl_source):
        """Write source to a temp file and consult it."""
        fd, path = tempfile.mkstemp(suffix='.pl', prefix=f'{self.module}_')
        self.pl_file = path
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(pl_source)
            results = list(self._engine.query(f"consult('{path}')"))
        except Exception as e:
            raise e

    # ------------------------------------------------------------------ #
    # Query execution                                                      #
    # ------------------------------------------------------------------ #

    def _query(self, goal):
        """Run a goal, catching all errors."""
        return list(self._engine.query(
            f"catch({self.module}:{goal}, _, fail)"
        ))

    def _parse_query(self, query):
        query = query.strip()
        # Strip leading negation: !Smart(Gary, True) -> Smart(Gary, False)
        negate = query.startswith('!')
        if negate:
            query = query[1:].strip()
        # Two-arg: Likes(Bear, Cat, True)
        m2 = re.match(r'(\w+)\(([^,]+),\s*([^,]+),\s*(True|False)\)', query)
        if m2:
            predicate = m2.group(1)
            subj1 = m2.group(2).strip().lower()
            subj2 = m2.group(3).strip().lower()
            value = (m2.group(4) == 'True') != negate  # flip if negated
            return predicate, (subj1, subj2), value
        # One-arg: Furry(Anne, True)
        m1 = re.match(r'(\w+)\(([^,]+),\s*(True|False)\)', query)
        if m1:
            value = (m1.group(3) == 'True') != negate
            return m1.group(1), m1.group(2).strip().lower(), value
        raise ValueError(f"Cannot parse query: {query}")

    def execute_program(self):
        """
        Open-World Assumption:
          TRUE provable   -> derived = True
          FALSE provable  -> derived = False
          Neither         -> derived = None  (Unknown)
        """
        try:
            predicate, subject, value_to_check = self._parse_query(self.Query[0])
            true_functor = f"{predicate.lower()}_true"
            false_functor = f"{predicate.lower()}_false"

            # subject is either a string (1-arg) or tuple of strings (2-arg)
            if isinstance(subject, tuple):
                args = ', '.join(subject)  # already lowercased in _parse_query
            else:
                args = subject  # already lowercased in _parse_query
            true_result = bool(self._query(f"{true_functor}({args})"))
            false_result = bool(self._query(f"{false_functor}({args})"))

            if true_result and not false_result:
                derived = True
            elif false_result and not true_result:
                derived = False
            elif true_result and false_result:
                derived = True   # contradiction
            else:
                derived = None   # unknown

            return self.answer_map[self.dataset_name](derived, value_to_check), ""
        except Exception as e:
            return None, str(e)
        finally:
            # Clean up temp file
            if self.pl_file and os.path.exists(self.pl_file):
                os.unlink(self.pl_file)
                self.pl_file = None

    def answer_mapping(self, answer):
        return answer

    def _answer_map_prontoqa(self, derived, vtc):
        return 'A' if derived == vtc else 'B'

    def _answer_map_proofwriter(self, derived, vtc):
        if derived is None:
            return 'C'
        return 'A' if derived == vtc else 'B'


# ------------------------------------------------------------------ #
# Smoke test                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cases = [
        ('A', 'ProofWriter', "Predicates:\nFurry($x, bool)\nNice($x, bool)\nFacts:\nFurry(Anne, True)\nRules:\nFurry($x, True) >>> Nice($x, True)\nQuery:\nNice(Anne, True)"),
        ('B', 'ProofWriter', "Predicates:\nFurry($x, bool)\nNice($x, bool)\nFacts:\nFurry(Anne, True)\nRules:\nFurry($x, True) >>> Nice($x, True)\nQuery:\nNice(Anne, False)"),
        ('C', 'ProofWriter', "Predicates:\nFurry($x, bool)\nNice($x, bool)\nFacts:\nFurry(Anne, True)\nRules:\nFurry($x, False) >>> Nice($x, True)\nQuery:\nNice(Anne, True)"),
        # Multi-hop
        ('A', 'ProntoQA', "Predicates:\nTumpus($x, bool)\nVumpus($x, bool)\nCold($x, bool)\nFacts:\nTumpus(Alex, True)\nRules:\nTumpus($x, True) >>> Vumpus($x, True)\nVumpus($x, True) >>> Cold($x, True)\nQuery:\nCold(Alex, True)"),
        # Circular rules - must not hang
        ('A', 'ProofWriter', "Predicates:\nRough($x, bool)\nSmart($x, bool)\nFurry($x, bool)\nFacts:\nRough(Bob, True)\nRules:\nRough($x, True) >>> Smart($x, True)\nSmart($x, True) >>> Furry($x, True)\nFurry($x, True) >>> Rough($x, True)\nQuery:\nSmart(Bob, True)"),
    ]

    import signal
    def handler(sig, frame): raise TimeoutError("Hung!")
    signal.signal(signal.SIGALRM, handler)

    all_pass = True
    for expected, dataset, lp in cases:
        signal.alarm(10)
        try:
            prog = Prolog_Program(lp, dataset)
            result, err = prog.execute_program()
            signal.alarm(0)
            ok = result == expected
            all_pass = all_pass and ok
            status = "OK  " if ok else "FAIL"
            print(f"[{status}] [{dataset}] Expected={expected} Got={result}"
                  + (f"  err={err}" if err else ""))
        except TimeoutError as e:
            signal.alarm(0)
            all_pass = False
            print(f"[HANG] [{dataset}] Expected={expected} — {e}")

    print("\nAll tests passed!" if all_pass else "\nSome tests FAILED.")
