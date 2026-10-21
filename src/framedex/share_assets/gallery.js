// fdx share gallery: soft fade-in as photos load, and a chrome-less viewer
// (arrows, swipe, Esc, tap to close). Locked galleries hand `fdx.load` an
// async loader that turns an <img> into a decrypted blob URL.
(function () {
  var d = document, fdx = (window.fdx = window.fdx || {});
  d.documentElement.classList.remove("no-js");

  fdx.reveal = function (root) {
    [].forEach.call(root.querySelectorAll(".ph img"), function (im) {
      function on() { im.classList.add("in"); }
      if (im.complete && im.naturalWidth) on();
      else { im.addEventListener("load", on); im.addEventListener("error", on); }
    });
  };

  var lb, big, links = [], at = 0, x0 = null, swiped = false, opener = null;
  function full(a) { return fdx.full ? fdx.full(a) : Promise.resolve(a.getAttribute("href")); }
  function show(n) {
    at = (n + links.length) % links.length;
    var a = links[at];
    big.alt = a.querySelector("img").alt;
    full(a).then(function (src) { if (links[at] === a) big.src = src; });
  }
  function open(n) {
    opener = links[n]; show(n);
    lb.classList.add("open"); lb.setAttribute("aria-hidden", "false");
    d.body.style.overflow = "hidden"; history.pushState({ fdxLb: 1 }, "");
    lb.querySelector(".x").focus({ preventScroll: true });
  }
  function shut() {
    lb.classList.remove("open"); lb.setAttribute("aria-hidden", "true");
    d.body.style.overflow = ""; big.removeAttribute("src");
    if (opener) opener.focus({ preventScroll: true });
  }
  function close() { if (history.state && history.state.fdxLb) history.back(); else shut(); }

  fdx.bind = function (root) {
    if (!lb) {
      lb = d.createElement("div"); lb.className = "lb"; lb.setAttribute("aria-hidden", "true");
      lb.setAttribute("role", "dialog"); lb.setAttribute("aria-label", "Photo");
      lb.innerHTML = '<img alt="">' +
        '<button class="prev" type="button" aria-label="Previous photo">&lsaquo;</button>' +
        '<button class="next" type="button" aria-label="Next photo">&rsaquo;</button>' +
        '<button class="x" type="button" aria-label="Close">&times;</button>';
      d.body.appendChild(lb); big = lb.querySelector("img");
      lb.addEventListener("click", function (e) {
        if (swiped) { swiped = false; return; }
        var c = e.target.classList;
        if (c.contains("prev")) show(at - 1);
        else if (c.contains("next")) show(at + 1);
        else close();
      });
      lb.addEventListener("touchstart", function (e) {
        x0 = e.touches.length === 1 ? e.touches[0].clientX : null; swiped = false;
      }, { passive: true });
      lb.addEventListener("touchend", function (e) {
        if (x0 === null) return;
        var dx = e.changedTouches[0].clientX - x0; x0 = null;
        if (Math.abs(dx) > 40) { swiped = true; show(at + (dx < 0 ? 1 : -1)); }
      });
      d.addEventListener("keydown", function (e) {
        if (!lb.classList.contains("open")) return;
        if (e.key === "ArrowRight") show(at + 1);
        else if (e.key === "ArrowLeft") show(at - 1);
        else if (e.key === "Escape") close();
      });
      addEventListener("popstate", function () { if (lb.classList.contains("open")) shut(); });
    }
    links = [].slice.call(root.querySelectorAll(".ph a"));
    links.forEach(function (a, n) {
      a.addEventListener("click", function (e) {
        if (fdx.editing || e.metaKey || e.ctrlKey || e.shiftKey) return;
        e.preventDefault(); open(n);
      });
    });
  };

  if (!fdx.locked) { fdx.reveal(d); fdx.bind(d); }
})();
