window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});

component$.subscribe(({ ref }) => {
  if (ref.classList.contains("md-annotation")) MathJax.typesetPromise([ref]);
});

// Link previews (`.md-tooltip2`) insert page content after the page is
// typeset, so typeset their math as it appears (once per element: typesetting
// itself adds nodes).
const typeset = new WeakSet();
let pending = 0;
new MutationObserver(() => {
  cancelAnimationFrame(pending);
  pending = requestAnimationFrame(() => {
    const math = [...document.querySelectorAll(".md-tooltip2 .arithmatex")].filter(
      (el) => !typeset.has(el),
    );
    if (!math.length) return;
    for (const el of math) typeset.add(el);
    MathJax.typesetPromise(math);
  });
}).observe(document.body, { childList: true, subtree: true });
