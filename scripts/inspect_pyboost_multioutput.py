import inspect


def _safe_sig(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def main() -> None:
    import py_boost

    print('py_boost version:', getattr(py_boost, '__version__', 'unknown'))

    try:
        import py_boost.multioutput as mo
    except Exception as e:
        print('Failed to import py_boost.multioutput:', repr(e))
        raise

    names = [n for n in dir(mo) if not n.startswith('_')]
    print('\npy_boost.multioutput public names:')
    for n in names:
        obj = getattr(mo, n)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print('-', n, '->', getattr(obj, '__module__', None), _safe_sig(obj))

    # Look for sketches / multioutput helpers
    key = [n for n in names if 'sketch' in n.lower() or 'multi' in n.lower() or 'output' in n.lower()]
    print('\nPotential multioutput-related symbols:', key)

    # Also check GradientBoosting params for multioutput support
    from py_boost import GradientBoosting
    print('\nGradientBoosting.__init__:', _safe_sig(GradientBoosting.__init__))


if __name__ == '__main__':
    main()
