import os
import re
import uuid
import tempfile
import hashlib
from pyswip import Prolog


class Prolog_Program:
    _engine = Prolog()  # one shared SWI-Prolog process
    _compiled_cache = {}  # cache compiled modules by source hash

    def __init__(self, logic_program, dataset_name='ProntoQA', timeout=60):
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.timeout = timeout
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
                program_str, segment = program_str.split(keyword, 1)  # Split on first occurrence only
                lines = [
                    l.split(':::')[0].strip()
                    for l in segment.strip().split('\n')
                    if l.strip() and not l.strip().startswith(':::')
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

    def _normalize_constant(self, name):
        """Normalize constant names: lowercase, spaces to underscores."""
        if not name:
            return name
        return name.strip().lower().replace(' ', '_')

    def _dsl_atom_to_prolog(self, atom):
        """
        Convert DSL atom to Prolog term.
        
        One-arg:  Furry(Anne, True)     ->  furry_true(anne)
                  Furry($x, True)       ->  furry_true(X)
        
        Two-arg:  Likes(Bear, Cat, True) -> likes_true(bear, cat)
                  Likes($x, $y, True)   -> likes_true(X, Y)
        
        Returns (prolog_term, was_negated) or None if parsing fails.
        """
        atom = atom.strip()
        negated = False
        
        # Check for outer negation (only at query time, not in rule bodies)
        # Rule body negation is handled separately in _build_pl_source
        
        # Try two-argument form first: Pred(arg1, arg2, True/False)
        m2 = re.match(r'(\w+)\(([^,]+),\s*([^,]+),\s*(True|False)\)', atom)
        if m2:
            predicate = m2.group(1).lower()
            arg1 = m2.group(2).strip()
            arg2 = m2.group(3).strip()
            value = m2.group(4).lower()
            
            def to_prolog_arg(a):
                if a.startswith('$'):
                    return a[1:].upper()
                else:
                    return self._normalize_constant(a)
            
            return f"{predicate}_{value}({to_prolog_arg(arg1)}, {to_prolog_arg(arg2)})"
        
        # Fall back to one-argument form: Pred(arg, True/False)
        m1 = re.match(r'(\w+)\(([^,]+),\s*(True|False)\)', atom)
        if m1:
            predicate = m1.group(1).lower()
            arg = m1.group(2).strip()
            value = m1.group(3).lower()
            prolog_arg = arg[1:].upper() if arg.startswith('$') else self._normalize_constant(arg)
            return f"{predicate}_{value}({prolog_arg})"
        
        return None

    def _get_functor_and_arity(self, term):
        """Extract (functor_name, arity) from a Prolog term string."""
        if term is None:
            return None
        # Handle possible negation wrapper
        clean_term = term
        if clean_term.startswith('\\+ '):
            clean_term = clean_term[3:]
        
        name = clean_term.split('(')[0]
        # Count arguments (arity)
        if '(' not in clean_term:
            return (name, 0)
        # Count commas between parentheses
        paren_content = clean_term[clean_term.index('(')+1:clean_term.rindex(')')]
        arity = paren_content.count(',') + 1 if paren_content else 1
        return (name, arity)

    def _collect_functors(self):
        """Collect all unique (functor, arity) pairs for tabling declarations."""
        functors = {}  # name -> arity
        
        def add_from_term(term):
            if term is None:
                return
            # Handle negated terms
            if isinstance(term, str) and term.startswith('\\+ '):
                term = term[3:]
            fa = self._get_functor_and_arity(term)
            if fa:
                functors[fa[0]] = fa[1]
        
        for rule in (self.Rules or []):
            if '>>>' not in rule:
                continue
            _, conclusion_str = rule.split('>>>')
            for c in conclusion_str.split('&&'):
                term = self._dsl_atom_to_prolog(c.strip())
                add_from_term(term)
        
        for fact in (self.Facts or []):
            term = self._dsl_atom_to_prolog(fact.strip())
            add_from_term(term)
        
        # Also collect from queries for completeness
        for query in (self.Query or []):
            try:
                pred, subject, _ = self._parse_query(query)
                # Add both true and false variants for the predicate
                functors[f"{pred.lower()}_true"] = 1 if isinstance(subject, str) else 2
                functors[f"{pred.lower()}_false"] = 1 if isinstance(subject, str) else 2
            except Exception:
                pass
        
        return functors

    def _translate_literal(self, literal):
        """
        Translate a single literal (possibly negated) to Prolog.
        Returns Prolog term string.
        """
        literal = literal.strip()
        negated = literal.startswith('!')
        if negated:
            literal = literal[1:].strip()
        
        term = self._dsl_atom_to_prolog(literal)
        if term is None:
            return None
        
        if negated:
            return f"\\+ {term}"
        return term

    def _build_pl_source(self):
        """Build the full Prolog source as a string."""
        lines = []

        # Module declaration
        lines.append(f':- module({self.module}, []).')
        lines.append('')

        # Table declarations for all derived predicates (prevents infinite loops)
        functors = self._collect_functors()
        for f in sorted(functors.keys()):
            arity = functors[f]
            lines.append(f':- table {f}/{arity}.')
        
        if functors:
            discontig = ', '.join(f'{f}/{functors[f]}' for f in sorted(functors.keys()))
            lines.append(f':- discontiguous {discontig}.')
        lines.append('')

        # Facts
        for fact in (self.Facts or []):
            fact = fact.strip()
            if not fact:
                continue
            # Skip facts with variables (shouldn't happen in well-formed input)
            if '$' in fact:
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
            
            # Parse premises with negation support
            premises = []
            for p in premise_str.split('&&'):
                p = p.strip()
                if not p:
                    continue
                prolog_premise = self._translate_literal(p)
                if prolog_premise is None:
                    break
                premises.append(prolog_premise)
            else:  # Only execute if no break occurred
                # Parse conclusions
                conclusions = []
                for c in conclusion_str.split('&&'):
                    c = c.strip()
                    if not c:
                        continue
                    prolog_conclusion = self._dsl_atom_to_prolog(c)
                    if prolog_conclusion is None:
                        break
                    conclusions.append(prolog_conclusion)
                else:
                    # Create rules
                    body = ', '.join(premises)
                    for head in conclusions:
                        lines.append(f'{head} :- {body}.')

        return '\n'.join(lines)

    def _consult(self, pl_source):
        """Write source to a temp file and consult it, with caching."""
        # Check cache first
        source_hash = hashlib.md5(pl_source.encode()).hexdigest()
        if source_hash in self._compiled_cache:
            # Reuse cached module - we need to copy the predicates
            # Since we can't easily copy modules, we'll still consult
            # but cache miss will be rare for identical rule sets
            pass
        
        fd, path = tempfile.mkstemp(suffix='.pl', prefix=f'{self.module}_')
        self.pl_file = path
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(pl_source)
            # Use call_with_time_limit for the consultation
            consult_query = f"catch((call_with_time_limit({self.timeout}, consult('{path}'))), _, fail)"
            results = list(self._engine.query(consult_query))
            # Cache the source for future use (optional)
            self._compiled_cache[source_hash] = pl_source
        except Exception as e:
            raise e

    # ------------------------------------------------------------------ #
    # Query execution                                                      #
    # ------------------------------------------------------------------ #

    def _query(self, goal):
        """Run a goal with timeout, catching all errors."""
        query = f"catch((call_with_time_limit({self.timeout}, {self.module}:{goal})), _, fail)"
        try:
            return list(self._engine.query(query))
        except Exception as e:
            print(f"[Prolog_Program] Query error: {e}")
            return []

    def _parse_query(self, query):
        """
        Parse a query string.
        Returns (predicate, subject, value_to_check)
        subject is either a string (1-arg) or tuple of strings (2-arg)
        """
        query = query.strip()
        # Strip leading negation: !Smart(Gary, True) -> Smart(Gary, False)
        negate = query.startswith('!')
        if negate:
            query = query[1:].strip()
        
        # Two-arg: Likes(Bear, Cat, True)
        m2 = re.match(r'(\w+)\(([^,]+),\s*([^,]+),\s*(True|False)\)', query)
        if m2:
            predicate = m2.group(1)
            subj1 = self._normalize_constant(m2.group(2).strip())
            subj2 = self._normalize_constant(m2.group(3).strip())
            value = (m2.group(4) == 'True') != negate  # flip if negated
            return predicate, (subj1, subj2), value
        
        # One-arg: Furry(Anne, True)
        m1 = re.match(r'(\w+)\(([^,]+),\s*(True|False)\)', query)
        if m1:
            predicate = m1.group(1)
            subject = self._normalize_constant(m1.group(2).strip())
            value = (m1.group(3) == 'True') != negate
            return predicate, subject, value
        
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

            # Format arguments for Prolog
            if isinstance(subject, tuple):
                args = ', '.join(subject)  # already normalized
            else:
                args = subject  # already normalized
            
            true_result = bool(self._query(f"{true_functor}({args})"))
            false_result = bool(self._query(f"{false_functor}({args})"))

            if true_result and not false_result:
                derived = True
            elif false_result and not true_result:
                derived = False
            elif true_result and false_result:
                # Contradiction - treat as True? Or log warning?
                # For ProofWriter, contradictions shouldn't happen with consistent data
                derived = True
            else:
                derived = None  # unknown

            return self.answer_map[self.dataset_name](derived, value_to_check), ""
        
        except Exception as e:
            return None, str(e)
        
        finally:
            # Clean up temp file
            if self.pl_file and os.path.exists(self.pl_file):
                try:
                    os.unlink(self.pl_file)
                except Exception:
                    pass
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
    import signal
    
    def handler(sig, frame):
        raise TimeoutError("Hung!")
    signal.signal(signal.SIGALRM, handler)
    
    print("Testing ProofWriter attribution example...")
    lp_att = """Predicates:
Cold($x, bool)
Quiet($x, bool)
Red($x, bool)
Smart($x, bool)
Round($x, bool)
Kind($x, bool)

Facts:
Cold(Bob, True)
Quiet(Bob, True)
Red(Bob, True)
Smart(Bob, True)
Kind(Charlie, True)

Rules:
Quiet($x, True) && Cold($x, True) >>> Smart($x, True)
Red($x, True) && Cold($x, True) >>> Round($x, True)

Query:
Kind(Charlie, True)"""
    
    signal.alarm(10)
    try:
        prog = Prolog_Program(lp_att, 'ProofWriter')
        result, err = prog.execute_program()
        signal.alarm(0)
        print(f"ProofWriter attribution test: Expected=A, Got={result}")
    except TimeoutError:
        signal.alarm(0)
        print("ProofWriter attribution test: TIMEOUT")
    
    print("\nTesting rule with negation in body...")
    lp_neg = """Predicates:
Sees($x, $y, bool)
Green($x, bool)
Visits($x, $y, bool)

Facts:
Sees(cat, squirrel, True)
Green(cat, False)

Rules:
Sees($x, $y, True) && !Green($x, True) >>> Visits($x, $y, True)

Query:
Visits(cat, squirrel, True)"""
    
    signal.alarm(10)
    try:
        prog = Prolog_Program(lp_neg, 'ProofWriter')
        result, err = prog.execute_program()
        signal.alarm(0)
        print(f"Negation test: Expected=A, Got={result}")
    except TimeoutError:
        signal.alarm(0)
        print("Negation test: TIMEOUT")
    
    print("\nTesting multi-word constants...")
    lp_multi = """Predicates:
Sees($x, $y, bool)
Green($x, bool)

Facts:
Sees(bald eagle, squirrel, True)

Rules:

Query:
Sees(bald eagle, squirrel, True)"""
    
    signal.alarm(10)
    try:
        prog = Prolog_Program(lp_multi, 'ProofWriter')
        result, err = prog.execute_program()
        signal.alarm(0)
        print(f"Multi-word test: Expected=A, Got={result}")
    except TimeoutError:
        signal.alarm(0)
        print("Multi-word test: TIMEOUT")
    
    print("\nTesting unknown case...")
    lp_unknown = """Predicates:
Furry($x, bool)

Facts:

Rules:

Query:
Furry(Bob, True)"""
    
    signal.alarm(10)
    try:
        prog = Prolog_Program(lp_unknown, 'ProofWriter')
        result, err = prog.execute_program()
        signal.alarm(0)
        print(f"Unknown test: Expected=C, Got={result}")
    except TimeoutError:
        signal.alarm(0)
        print("Unknown test: TIMEOUT")