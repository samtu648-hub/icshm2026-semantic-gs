#!/usr/bin/env node
// Exercise the real offline deliverable in a fresh browser profile.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const opt=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,arr)=>i%2?a:[...a,[v.replace(/^--/,''),arr[i+1]]],[]));
const require=createRequire(import.meta.url);
const {chromium}=require(path.join(opt['node-modules'],'playwright'));
const report=path.resolve(opt.report),qa=path.resolve(opt.qa);fs.mkdirSync(qa,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:opt.chrome||'/usr/bin/google-chrome',args:['--no-sandbox','--disable-gpu']});
const errors=[],network=[],checks={};
try {
 const page=await browser.newPage({viewport:{width:1440,height:1080}});
 page.on('pageerror',e=>errors.push(String(e)));
 await page.route('**/*',route=>{if(/^https?:/.test(route.request().url())){network.push(route.request().url());return route.abort()}return route.continue()});
 await page.goto(pathToFileURL(path.join(report,'index.html')).href);
 await page.waitForFunction(()=>window.reportReady===true);
 await page.locator('#best-score').waitFor();
 checks.best_score=await page.locator('#best-score').innerText();
 if(checks.best_score!=='83.705003%')throw Error('Primary score differs');
 await page.evaluate(async()=>{for(const i of document.images){i.loading='eager';if(!i.complete)await new Promise(r=>{i.onload=r;i.onerror=r})}});
 checks.images=await page.evaluate(()=>Array.from(document.images).map(i=>({src:i.getAttribute('src'),width:i.naturalWidth})));
 if(checks.images.some(x=>!x.width))throw Error('Missing image');
 await page.screenshot({path:path.join(qa,'desktop-hero.png')});
 await page.locator('#routes').screenshot({path:path.join(qa,'routes.png')});
 await page.locator('#gallery').scrollIntoViewIfNeeded();
 for(const id of ['054','081','140','159','160','183','180','145','088','061']){
   await page.locator(`#samples button[data-id="${id}"]`).click();
   if(!(await page.locator('#sample-note').innerText()).startsWith(id))throw Error('Gallery selection failed');
 }
 await page.locator('#samples button[data-id="183"]').click();
 await page.locator('#view-tabs button[data-mode="history"]').click();
 await page.locator('#history-version').selectOption('sem111');
 if(!await page.locator('#viewer img[data-key="sem111"]').count())throw Error('Historical version not loaded');
 await page.locator('#view-tabs button[data-mode="rgb"]').click();
 if(!await page.locator('#viewer img[data-key="standard"]').count())throw Error('RGB comparison failed');
 await page.locator('#view-tabs button[data-mode="latest"]').click();
 await page.locator('#viewer img').first().click({position:{x:100,y:80}});
 await page.locator('#zoom').fill('8');
 await page.waitForFunction(()=>document.getElementById('zoom-value').textContent==='8×');
 checks.gallery_modes_and_zoom=true;
 await page.locator('#gallery').screenshot({path:path.join(qa,'gallery.png')});
 await page.locator('#search').fill('SEM329');
 const matched=await page.locator('#registry-list .exp').allTextContents();
 if(!matched.includes('SEM-329-SECOND-SOURCE-VETO-FINAL'))throw Error('Registry search failed');
 await page.locator('#registry-list > details > summary').first().click();
 checks.registry_search=true;
 await page.locator('#search').fill('');
 if(await page.locator('video').count()!==4)throw Error('Four current videos required');
 checks.videos=[];
 for(let i=0;i<4;i++){
  const v=page.locator(`#video-${i}`);await v.scrollIntoViewIfNeeded();
  await v.evaluate(async el=>{
   el.load();
   await new Promise((resolve,reject)=>{
    if(el.readyState>=1)return resolve();
    el.addEventListener('loadedmetadata',resolve,{once:true});
    el.addEventListener('error',()=>reject(Error('Video decode error')),{once:true});
   });
  });
  await page.locator(`button[data-video="video-${i}"][data-rate="0.5"]`).click();
  const state=await v.evaluate(async el=>{el.currentTime=3;await el.play();await new Promise(r=>setTimeout(r,150));el.pause();return {duration:el.duration,currentTime:el.currentTime,rate:el.playbackRate,width:el.videoWidth,height:el.videoHeight,error:el.error?.message||null}});
  if(state.duration!==6||state.rate!==.5||state.currentTime<=3||state.error)throw Error('Playback / seek / rate failure');checks.videos.push(state);
 }
 await page.locator('#videos').screenshot({path:path.join(qa,'videos.png')});
 checks.local_links=[];
 const links=await page.locator('a[href]').evaluateAll(es=>es.map(e=>e.getAttribute('href')));
 for(const href of new Set(links)){if(href.startsWith('#'))continue;if(/^https?:/.test(href))throw Error('External report dependency');const target=path.resolve(report,decodeURIComponent(href.split('#')[0]));if(!fs.existsSync(target))throw Error('Broken local link: '+href);checks.local_links.push(href)}
 for(const width of [390,320]){
  await page.setViewportSize({width,height:844});await page.goto(pathToFileURL(path.join(report,'index.html')).href);await page.waitForFunction(()=>window.reportReady===true);
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);if(overflow)throw Error('Horizontal overflow at '+width);
  await page.screenshot({path:path.join(qa,`mobile-${width}.png`)});
  await page.locator('#gallery').screenshot({path:path.join(qa,`gallery-${width}.png`)});
 }
 checks.widths=[1440,390,320];
 if(errors.length||network.length)throw Error('Unexpected JS errors or external network requests');
 checks.errors=errors;checks.external_requests=network;checks.status='passed_offline_browser_interactions';
 fs.writeFileSync(path.join(qa,'browser-audit.json'),JSON.stringify(checks,null,2)+'\n');
 console.log(JSON.stringify({status:checks.status,images:checks.images.length,videos:checks.videos.length,local_links:checks.local_links.length,widths:checks.widths}));
} finally {await browser.close()}
