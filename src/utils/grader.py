import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def verify_math(student_input, true_answer):
    """
    Safely converts strings to math objects and checks absolute equivalence.
    Returns True if mathematically equal, False otherwise.
    """
    try:
        # 1. Clean up common text additions (e.g., "x =", "y=", "cm")
        student_clean = student_input.lower().replace("x", "").replace("=", "").replace("cm", "").strip()
        true_clean = str(true_answer).lower().replace("x", "").replace("=", "").replace("cm", "").strip()

        # 2. Allow SymPy to understand implicit multiplication (e.g., 2x -> 2*x)
        transformations = standard_transformations + (implicit_multiplication_application,)

        # 3. Parse strings into SymPy mathematical expressions
        student_expr = parse_expr(student_clean, transformations=transformations)
        true_expr = parse_expr(true_clean, transformations=transformations)

        # 4. Check if the mathematical difference is zero
        return sympy.simplify(student_expr - true_expr) == 0
        
    except Exception as e:
        # If the student typed gibberish or full text that SymPy can't parse, fail them safely.
        print(f"SymPy parse error: {e}")
        return False
