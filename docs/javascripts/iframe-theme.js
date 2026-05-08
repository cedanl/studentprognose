// Synchronise Material for MkDocs dark-mode toggle with embedded Plotly iframes.
function broadcastTheme() {
  var scheme =
    document.body.getAttribute("data-md-color-scheme") || "default";
  document.querySelectorAll("iframe").forEach(function (iframe) {
    try {
      iframe.contentWindow.postMessage({ type: "theme", scheme: scheme }, "*");
    } catch (_) {
      /* cross-origin iframe, ignore */
    }
  });
}

new MutationObserver(function (mutations) {
  for (var i = 0; i < mutations.length; i++) {
    if (mutations[i].attributeName === "data-md-color-scheme") {
      setTimeout(broadcastTheme, 100);
    }
  }
}).observe(document.body, { attributes: true });

document.addEventListener("DOMContentLoaded", function () {
  setTimeout(broadcastTheme, 500);
});
