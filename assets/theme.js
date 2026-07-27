/* Light/dark toggle for the landing page.
   Light is the default, so an absent data-theme means light rather than
   "follow the OS" -- unlike the visualization page, which defers to the OS. */
(function () {
    "use strict";

    var root = document.documentElement;

    function isDark() {
        return root.getAttribute("data-theme") === "dark";
    }

    document.getElementById("theme").addEventListener("click", function () {
        root.setAttribute("data-theme", isDark() ? "light" : "dark");
    });
})();
