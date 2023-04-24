import typer
from enum import Enum
from rich import print

import numpy as np
from linear_algebra.algebra.cramer import cramer
from linear_algebra.algebra.lup import lup_factorization

from linear_algebra.algebra.util import inv, rank


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

            print(f'[green]INFO[/green]: Solution vector: ')
            print(x)
        case 'schur':
            ...


if __name__ == '__main__':
    typer.run(main)
