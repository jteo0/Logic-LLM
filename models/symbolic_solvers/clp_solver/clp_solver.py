import os
import re
import uuid
import tempfile
from collections import defaultdict
from pyswip import Prolog


class CLP_Program:
    _engine = Prolog()

    def __init__(self, logic_program: str, dataset_name: str) -> None:
        self.logic_program = logic_program
        self.dataset_name = dataset_name
        self.flag = self._parse_logic_program()

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_logic_program(self):
        keywords = ['Query:', 'Constraints:', 'Variables:', 'Domain:']
        program_str = self.logic_program
        for keyword in keywords:
            try:
                program_str, segment = program_str.split(keyword)
                lines = [
                    l.split(':::')[0].strip()
                    for l in segment.strip().split('\n')
                    if l.strip()
                ]
                setattr(self, keyword[:-1], lines)
            except Exception:
                setattr(self, keyword[:-1], None)

        return all(
            getattr(self, k, None) is not None
            for k in ['Query', 'Constraints', 'Variables', 'Domain']
        )

    # ------------------------------------------------------------------ #
    # DSL -> CLP(FD) translation                                          #
    # ------------------------------------------------------------------ #

    def _parse_variables(self):
        """
        Parse variable declarations.
        'green_book [IN] [1, 2, 3, 4, 5]'  ->  ('green_book', 1, 5)
        Returns list of (prolog_name, lo, hi) tuples.
        """
        vars_info = []
        for v in self.Variables:
            m = re.match(r'(\S+)\s*\[IN\]\s*\[(.+)\]', v)
            if not m:
                continue
            name = m.group(1).replace(' ', '_').lower()
            values = [int(x.strip()) for x in m.group(2).split(',')]
            vars_info.append((name, min(values), max(values)))
        return vars_info

    def _translate_constraint(self, constraint: str, var_names: set) -> str | None:
        """
        Translate a single constraint string to CLP(FD) syntax.

        AllDifferentConstraint([a, b, c])  ->  all_different([A, B, C])
        a > b                              ->  A #> B
        a == 3                             ->  A #= 3
        a != b                             ->  A #\\= B
        a < b                              ->  A #< B
        a >= b                             ->  A #>= B
        a <= b                             ->  A #=< B
        """
        constraint = constraint.strip()

        # AllDifferentConstraint([var1, var2, ...])
        m = re.match(r'AllDifferentConstraint\(\[(.*?)\]\)', constraint)
        if m:
            vars_str = m.group(1)
            prolog_vars = ', '.join(
                v.strip().replace(' ', '_').capitalize()
                for v in vars_str.split(',')
            )
            return f"all_different([{prolog_vars}])"

        # Numeric constraints — replace operators with CLP(FD) equivalents
        # and variable names with Prolog variables (capitalized)
        result = constraint

        # Replace operators (order matters — do != and <= before = and <)
        result = result.replace('!=', '#\\=')
        result = result.replace('<=', '#=<')
        result = result.replace('>=', '#>=')
        result = result.replace('==', '#=')
        result = result.replace('<',  '#<')
        result = result.replace('>',  '#>')

        # Replace variable names with capitalized Prolog variables
        for var in sorted(var_names, key=len, reverse=True):
            prolog_var = var.replace(' ', '_').capitalize()
            result = re.sub(rf'\b{re.escape(var)}\b', prolog_var, result)

        return result

    def _build_pl_source(self, vars_info, var_names):
        """Build the full CLP(FD) Prolog source."""
        lines = [':- use_module(library(clpfd)).', '']

        # Build solve/N head with all variables as arguments
        prolog_vars = [name.capitalize() for name, _, _ in vars_info]
        args = ', '.join(prolog_vars)
        lines.append(f'solve({args}) :-')

        # Domain declarations
        for name, lo, hi in vars_info:
            pvar = name.capitalize()
            lines.append(f'    {pvar} in {lo}..{hi},')

        # Constraints
        clp_constraints = []
        for c in self.Constraints:
            translated = self._translate_constraint(c, var_names)
            if translated:
                clp_constraints.append(translated)

        # All constraints get a trailing comma since label() always follows
        for c in clp_constraints:
            lines.append(f'    {c},')

        # Labeling — find all solutions (last goal, no trailing comma)
        lines.append(f'    label([{args}]).')

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Execution                                                            #
    # ------------------------------------------------------------------ #

    def execute_program(self):
        """
        Build and run the CLP(FD) program.
        Returns (solutions, error_message) where solutions is a list of dicts
        mapping variable names to values — same format as python-constraint.
        """
        if not self.flag:
            return None, "Failed to parse logic program"

        try:
            vars_info = self._parse_variables()
            if not vars_info:
                return None, "No variables found"

            var_names = {name for name, _, _ in vars_info}
            pl_source = self._build_pl_source(vars_info, var_names)

            # Write to temp file and consult
            fd, path = tempfile.mkstemp(suffix='.pl', prefix=f'clp_{uuid.uuid4().hex}_')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(pl_source)
                list(self._engine.query(f"consult('{path}')"))
            finally:
                if os.path.exists(path):
                    os.unlink(path)

            # Query for all solutions
            prolog_vars = [name.capitalize() for name, _, _ in vars_info]
            args = ', '.join(prolog_vars)
            raw_solutions = list(self._engine.query(f'solve({args})'))

            # Convert to python-constraint format: [{var_name: value, ...}]
            solutions = []
            for sol in raw_solutions:
                solution = {}
                for name, _, _ in vars_info:
                    pvar = name.capitalize()
                    if pvar in sol:
                        solution[name] = sol[pvar]
                solutions.append(solution)

            return solutions, ""

        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #
    # Answer mapping (identical to original CSP_Program)                  #
    # ------------------------------------------------------------------ #

    def answer_mapping(self, answer):
        option_pattern = r'^\w+\)'
        expression_pattern = r'\w[\w\s]* == \d+'

        variable_ans_map = defaultdict(set)
        for result in answer:
            for variable, value in result.items():
                variable_ans_map[variable].add(value)

        for option_str in self.Query:
            option_match = re.match(option_pattern, option_str)
            if not option_match:
                continue
            option = option_match.group().replace(')', '')
            expression_match = re.search(expression_pattern, option_str)
            if not expression_match:
                continue
            expression_str = expression_match.group()
            variable, value = expression_str.split('==')
            variable = variable.strip().replace(' ', '_').lower()
            value = int(value.strip())
            if len(variable_ans_map[variable]) == 1 and value in variable_ans_map[variable]:
                return option

        return None


# ------------------------------------------------------------------ #
# Smoke test                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    lp = """Domain:
1: leftmost
5: rightmost
Variables:
green_book [IN] [1, 2, 3, 4, 5]
blue_book [IN] [1, 2, 3, 4, 5]
white_book [IN] [1, 2, 3, 4, 5]
purple_book [IN] [1, 2, 3, 4, 5]
yellow_book [IN] [1, 2, 3, 4, 5]
Constraints:
blue_book > yellow_book ::: The blue book is to the right of the yellow book.
white_book < yellow_book ::: The white book is to the left of the yellow book.
blue_book == 4 ::: The blue book is the second from the right.
purple_book == 2 ::: The purple book is the second from the left.
AllDifferentConstraint([green_book, blue_book, white_book, purple_book, yellow_book]) ::: All books have different values.
Query:
A) green_book == 2 ::: The green book is the second from the left.
B) blue_book == 2 ::: The blue book is the second from the left.
C) white_book == 2 ::: The white book is the second from the left.
D) purple_book == 2 ::: The purple book is the second from the left.
E) yellow_book == 2 ::: The yellow book is the second from the left."""

    prog = CLP_Program(lp, 'LogicalDeduction')
    ans, err = prog.execute_program()
    print('Solutions:', ans)
    result = prog.answer_mapping(ans)
    print('Answer:', result, '(expected: D)')
    print('OK!' if result == 'D' else 'FAIL')
