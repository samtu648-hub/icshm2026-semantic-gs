(() => {
  const search = document.querySelector('#search');
  const state = document.querySelector('#state');
  const rows = [...document.querySelectorAll('#experiment-table tbody tr')];
  const texts = rows.map(row => row.textContent.toLowerCase());
  const filter = () => {
    const query = search.value.trim().toLowerCase();
    let count = 0;
    rows.forEach((row, i) => {
      const show = (!query || texts[i].includes(query)) && (!state.value || row.dataset.state === state.value);
      row.hidden = !show;
      if (show) count += 1;
    });
    document.querySelector('#count').textContent = `${count} / ${rows.length}`;
    document.querySelector('#empty').hidden = count !== 0;
  };
  search.addEventListener('input', filter);
  state.addEventListener('change', filter);
  document.querySelectorAll('video').forEach(video => video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
  }));
})();
