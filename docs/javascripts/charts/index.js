// Mounts the interactive charts.  A chart is a placeholder in the markdown,
//
//   <div class="ms-chart" data-chart="spectra"></div>
//
// and a module `./spectra.js` whose default export `(el) => void` renders into it.
// Zensical's instant navigation swaps page content without reloading, so mount on
// every `document$` emission, not just once.

const mounted = new WeakSet();

function mountAll() {
  for (const el of document.querySelectorAll(".ms-chart[data-chart]")) {
    if (mounted.has(el)) continue;
    mounted.add(el);
    import(`./${el.dataset.chart}.js`)
      .then((module) => module.default(el))
      .catch((err) => {
        el.classList.add("ms-chart--error");
        el.textContent = "This interactive chart failed to load.";
        console.error(`chart ${el.dataset.chart}:`, err);
      });
  }
}

if (window.document$) window.document$.subscribe(mountAll);
else mountAll();
