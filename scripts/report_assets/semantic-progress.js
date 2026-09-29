(() => {
  const journey = document.querySelector('#journey-video');
  document.querySelectorAll('.stage').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.stage').forEach(x => x.classList.remove('active'));
    button.classList.add('active');
    journey.pause(); journey.poster = button.dataset.poster;
    journey.querySelector('source').src = button.dataset.src; journey.load();
    document.querySelector('#journey-title').textContent = button.dataset.title;
    document.querySelector('#journey-note').textContent = button.dataset.note;
  }));
  document.querySelectorAll('.example-tabs button').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.example-tabs button').forEach(x => x.classList.remove('active'));
    button.classList.add('active');
    const view = button.dataset.view;
    document.querySelectorAll('#example-grid img').forEach(img => {
      const kind = img.dataset.kind;
      img.src = `assets/examples/${view}-${kind}.${kind === 'photo' || kind === 'rgb' ? 'jpg' : 'png'}`;
    });
  }));
  document.querySelector('#present').addEventListener('click', async () => {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  });
  const top = document.querySelector('#to-top');
  addEventListener('scroll', () => top.classList.toggle('show', scrollY > innerHeight));
  top.addEventListener('click', () => scrollTo({top:0, behavior:'smooth'}));
  addEventListener('keydown', event => {
    if (!['ArrowDown','ArrowRight','ArrowUp','ArrowLeft'].includes(event.key)) return;
    if (event.target.matches('video,button,summary')) return;
    const sections = [...document.querySelectorAll('main>section')];
    const current = sections.reduce((best, s, i) => Math.abs(s.getBoundingClientRect().top - 64) < best.d ? {i,d:Math.abs(s.getBoundingClientRect().top - 64)} : best, {i:0,d:Infinity}).i;
    const delta = event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1;
    sections[Math.max(0, Math.min(sections.length - 1, current + delta))].scrollIntoView({behavior:'smooth'});
  });
})();
