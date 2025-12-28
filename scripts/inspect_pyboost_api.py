import inspect


def main() -> None:
    import py_boost

    print("py_boost version:", getattr(py_boost, "__version__", "unknown"))
    try:
        from py_boost import GradientBoosting
    except Exception as e:
        print("Failed to import GradientBoosting:", repr(e))
        raise

    print("GradientBoosting module:", GradientBoosting.__module__)
    print("GradientBoosting.__init__:", inspect.signature(GradientBoosting.__init__))
    print("GradientBoosting.fit:", inspect.signature(GradientBoosting.fit))

    try:
        print("GradientBoosting.predict:", inspect.signature(GradientBoosting.predict))
    except Exception:
        pass
    try:
        print("GradientBoosting.quantize:", inspect.signature(GradientBoosting.quantize))
    except Exception:
        pass

    fit_doc = getattr(GradientBoosting.fit, "__doc__", None)
    if fit_doc:
        print("\n--- GradientBoosting.fit docstring (head) ---")
        print("\n".join(fit_doc.strip().splitlines()[:40]))

    try:
        src = inspect.getsource(GradientBoosting.fit)
        print("\n--- GradientBoosting.fit source excerpt ---")
        print("\n".join(src.splitlines()[:120]))
    except OSError:
        # Could be a compiled / dynamically generated function.
        pass

    for fn_name in ["predict", "quantize"]:
        fn = getattr(GradientBoosting, fn_name, None)
        if not callable(fn):
            continue
        try:
            src = inspect.getsource(fn)
            print(f"\n--- GradientBoosting.{fn_name} source excerpt ---")
            print("\n".join(src.splitlines()[:120]))
        except OSError:
            pass


if __name__ == "__main__":
    main()
