// Forall Labs project pages: the shared interactive layer (pairs with fl-page.css).
// The same file ships in every project repository; edit it in one and copy it to the others.
(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- compare sliders ----------
  // <div class="cmp" data-sweep> <img class="cmp-a"> <img class="cmp-b"> labels <input type=range> <span class="cmp-handle"><span></span></span> </div>
  // cmp-b (left of the handle) is drawn over cmp-a.
  function initCompare(el) {
    const input = el.querySelector("input[type=range]");
    const set = (v) => { el.style.setProperty("--pos", v + "%"); input.value = v; };
    let touched = false;
    input.addEventListener("input", () => { touched = true; set(+input.value); });
    el.addEventListener("pointerdown", () => { touched = true; el.classList.add("dragging"); });
    ["pointerup", "pointercancel", "pointerleave"].forEach((ev) => el.addEventListener(ev, () => el.classList.remove("dragging")));
    set(+input.value || 50);
    // one hint sweep when the slider first comes into view, so it reads as draggable
    if (el.hasAttribute("data-sweep") && !reduced && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (!entries[0].isIntersecting) return;
        io.disconnect();
        const keys = [[0, 50], [650, 74], [1400, 30], [2100, 50]];
        const t0 = performance.now() + 700;
        const ease = (x) => x < .5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
        const frame = (now) => {
          if (touched) return;
          const t = now - t0;
          if (t < 0) return requestAnimationFrame(frame);
          let i = keys.findIndex((k) => k[0] > t);
          if (i === -1) return set(50);
          const [ta, va] = keys[i - 1], [tb, vb] = keys[i];
          set(va + (vb - va) * ease((t - ta) / (tb - ta)));
          requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
      }, { threshold: .5 });
      io.observe(el);
    }
  }
  document.querySelectorAll(".cmp").forEach(initCompare);
  // swap both pictures of a slider: flCompare(el, {a, b, aAlt, bAlt, aLabel, bLabel})
  window.flCompare = (el, o) => {
    const a = el.querySelector(".cmp-a"), b = el.querySelector(".cmp-b");
    if (o.a) { a.src = o.a; a.alt = o.aAlt || ""; }
    if (o.b) { b.src = o.b; b.alt = o.bAlt || ""; }
    if (o.aLabel != null) el.querySelector(".cmp-label.r").textContent = o.aLabel;
    if (o.bLabel != null) el.querySelector(".cmp-label.l").textContent = o.bLabel;
  };

  // ---------- section bar: mark the section in view ----------
  const links = [...document.querySelectorAll(".toc a[href^='#']")];
  const targets = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  if (targets.length) {
    const mark = () => {
      const line = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-h")) + 140;
      let cur = null;
      targets.forEach((t) => { if (t.getBoundingClientRect().top <= line) cur = t; });
      links.forEach((a) => a.setAttribute("aria-current", String(cur && a.getAttribute("href") === "#" + cur.id)));
    };
    let raf = 0;
    window.addEventListener("scroll", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(mark); }, { passive: true });
    mark();
  }

  // ---------- figure zoom: click a static figure to see it full size ----------
  const imgs = document.querySelectorAll(".plate img");
  if (imgs.length && "HTMLDialogElement" in window) {
    const dlg = document.createElement("dialog");
    dlg.className = "zoom-dlg";
    dlg.innerHTML = `<img alt=""><form method="dialog"><button class="btn" aria-label="Close the enlarged figure">&times;</button></form>`;
    document.body.appendChild(dlg);
    const big = dlg.querySelector("img");
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    imgs.forEach((img) => {
      img.setAttribute("data-zoom", "");
      img.setAttribute("tabindex", "0");
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", "Enlarge: " + (img.alt || "figure"));
      const open = () => { big.src = img.currentSrc || img.src; big.alt = img.alt; dlg.showModal(); };
      img.addEventListener("click", open);
      img.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });
  }
})();
