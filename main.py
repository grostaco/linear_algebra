import typer
from enum import Enum
from rich import print

import numpy as np
from linear_algebra.algebra.cramer import cramer
from linear_algebra.algebra.gaussian_elim import gaussian_elim
from linear_algebra.algebra.lup import lup_factorization
from linear_algebra.algebra.schur import schur

from linear_algebra.algebra.util import back_substitution, inv, rank


class SolverType(str, Enum):
    lup = 'lup'
    cramer = 'cramer'
    schur = 'schur'


def main(solver: SolverType = typer.Option(..., help='Solver type to be used'),
         in_file: str = typer.Argument(..., help='Input numpy dump file to be solved')):
    print(f':wrench: solver selected: [bold blue]{solver}[/bold blue]')
    print(f':file_folder: input file: [bold blue]{in_file}[/bold blue]\n')

    mat = np.load(in_file, allow_pickle=True)
    coef = mat[:, :-1]
    b = mat[:, -1:]
    print(
        f'[green]INFO[/green]: loaded matrix from file [bold blue]{in_file}[/bold blue]\n')
    print(f'[green]INFO[/green]: loaded coefficients matrix:')
    print(f'{coef}')
    print(f'[green]INFO[/green]: loaded constants matrix:')
    print(f'{b}\n')

    coef_mask = ~coef.any(-1)
    b_mask = ~b.any(-1)

    mask = coef_mask & b_mask
    coef = coef[~mask]
    b = b[~mask]

    print(f'[green]INFO[/green]: Truncating matrix')
    print(f'[green]INFO[/green]: Truncated coefficients matrix:')
    print(coef)
    print(f'[green]INFO[/green]: Truncated constants matrix: ')
    print(b)

    print(f'[green]INFO[/green]: solving...')

    n, m = coef.shape
    match solver:
        case 'lup' | 'cramer':
            if n != m:
                print(
                    f'[red]ERROR[/red]: solver type [blue]{solver}[/blue] cannot handle non-square matrices. Aborting')
                exit(1)

            if rank(mat) != n:
                print(
                    f'[red]ERROR[/red]: solver type {solver} cannot handle non-singular matrices. '
                    f'The input coefficients matrix either has infinite solutions or is inconsistent. Aborting')
                exit(1)

            if solver == 'lup':
                l, u, p = lup_factorization(coef)
                y = inv(l) @ p @ b
                x = inv(u) @ y
            else:
                x = cramer(coef, b.reshape(-1))
                x = np.array(list(x)).reshape(-1, 1)

        case 'schur':
            try:
                m = schur(np.hstack((coef, b)))
            except ValueError:
                m = gaussian_elim(np.hstack((coef, b)))

            ref, b = m[:, :-1], m[:, -1:]

            idx = ref.any(-1)

            ref_mask = ~ref.any(-1)
            b_mask = ~b.any(-1)

            mask = ref_mask & b_mask
            ref = ref[~mask]
            b = b[~mask]

            rm, rn = ref.shape

            if rm != rn:  # and not np.diag(ref).all():
                # Check if it is inconsistent
                if not (b[~idx] == 0).all():
                    print(f'[red]ERROR[/red]: The system is inconsistent. Aborting')
                    exit(1)
                else:
                    print(ref)
                    free_indices = []
                    for x in ref:
                        free_indices.append(np.nonzero(x != 0)[0][0])

                    print(free_indices)
                    print(
                        f'[green]INFO[/green]: The system is has infinitely many solutions. The possible free variables are: \n'
                        f'{set(range(rn)) - set(free_indices)}')
                    exit(0)
            else:
                x = back_substitution(ref[idx], b[idx])
        case _:
            raise ValueError()

    print(f'[green]INFO[/green]: Solution vector: ')
    print(x)


if __name__ == '__main__':
    typer.run(main)
