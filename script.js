const year = document.querySelector("#year");
if (year) {
  year.textContent = new Date().getFullYear();
}


const navLinks = Array.from(document.querySelectorAll(".site-nav a"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const setActiveLink = () => {
  const current = sections.findLast((section) => section.getBoundingClientRect().top < 140);
  navLinks.forEach((link) => {
    const isActive = current && link.getAttribute("href") === `#${current.id}`;
    link.toggleAttribute("aria-current", isActive);
  });
};

setActiveLink();
window.addEventListener("scroll", setActiveLink, { passive: true });
