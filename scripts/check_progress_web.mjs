#!/usr/bin/env node
// Real file://, offline UI checks; optional runtime supplied by caller.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const options=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,r)=>i%2?a:[...a,[v.replace(/^--/,''),r[i+1]]],[]));
const require=createRequire(import.meta.url);
const {chromium}=require(path.join(options['node-modules'],'playwright'));
const report=path.resolve(options.report),qa=path.resolve(options.qa);
fs.mkdirSync(qa,{recursive:true});
const browser=await chromium.launch({executablePath:options.chrome||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu']});
const context=await browser.newContext({offline:true,viewport:{width:1440,height:1000}});
const page=await context.newPage();const errors=[],network=[],checks={};
page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
const assert=(condition,label)=>{if(!condition)throw Error(label);checks[label]=true};
try{
 await page.goto(pathToFileURL(path.join(report,'index.html')).href);
 await page.waitForFunction(()=>document.querySelectorAll('[role=tab]').length===6);
 assert(await page.locator('#stage-title').textContent()==='全量训练最高分方案','default_best_stage');
 assert(await page.locator('#stage-video').evaluate(v=>v.paused),'no_autoplay');
 for(const size of [{width:1440,height:1000},{width:1366,height:768},{width:1920,height:1080},{width:390,height:844}]){
  await page.setViewportSize(size);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`no_overflow_${size.width}`);
  for(const id of ['evolution','milestones','metrics','difficult']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();
   if(size.width===1440||size.width===390)await page.locator('#'+id).screenshot({path:path.join(qa,`${size.width}-${id}.png`)});
  }
 }
 await page.setViewportSize({width:1440,height:1000});
 for(let i=0;i<6;i++){
  await page.locator(`#tab-${i}`).click();const v=page.locator('#stage-video');
  await v.evaluate(async x=>{await x.play()});await page.waitForTimeout(180);
  assert(await v.evaluate(x=>x.videoWidth===2640&&x.videoHeight===990&&x.currentTime>0&&!x.paused),`video_${i}_decodes_and_plays`);
  await v.evaluate(x=>{x.pause();x.currentTime=8});
  await page.waitForFunction(()=>Math.abs(document.querySelector('video').currentTime-8)<.1&&!document.querySelector('video').seeking);
  assert(await v.evaluate(x=>Math.abs(x.duration-12)<.01),`video_${i}_duration_and_seek`);
 }
 await page.locator('#video-fullscreen').click();assert(await page.evaluate(()=>!!document.fullscreenElement),'video_fullscreen');await page.evaluate(()=>document.exitFullscreen());
 await page.locator('#tab-5').focus();await page.keyboard.press('Home');assert(await page.locator('#tab-0').getAttribute('aria-selected')==='true','tab_home');await page.keyboard.press('ArrowDown');assert(await page.locator('#tab-1').getAttribute('aria-selected')==='true','tab_arrow');await page.keyboard.press('End');assert(await page.locator('#tab-5').getAttribute('aria-selected')==='true','tab_end');
 await page.locator('[data-stage="4"]').click();assert(await page.locator('#tab-4').getAttribute('aria-selected')==='true','flow_to_stage');
 for(let view=0;view<4;view++){
  await page.locator(`[data-view="${view}"]`).click();
  for(const mode of ['full','crop']){
   await page.locator(`#${mode}-view`).click();
   await page.waitForFunction(()=>[...document.querySelectorAll('#comparison img')].every(x=>x.complete&&x.naturalWidth>0));
   assert(await page.locator('#comparison img').evaluateAll(xs=>xs.length===4&&xs.every(x=>x.naturalWidth===xs[0].naturalWidth&&x.naturalHeight===xs[0].naturalHeight)),`view_${view}_${mode}_aligned`);
  }
 }
 await page.locator('#comparison button').first().click();assert(await page.locator('#image-dialog').evaluate(d=>d.open),'image_zoom');await page.keyboard.press('Escape');assert(await page.locator('#image-dialog').evaluate(d=>!d.open),'image_escape_close');
 for(let i=0;i<3;i++){await page.locator('#trajectory-details button').nth(i).click();assert(await page.locator('#image-dialog img').evaluate(x=>x.complete&&x.naturalHeight===1978),`trajectory_detail_${i}`);await page.locator('#close-image').click();}
 await page.locator('.rgb-details summary').click();assert(await page.locator('#rgb-full table').isVisible(),'rgb_metrics_expand');
 await page.locator('#sources summary').click();await page.locator('#source-list a').first().click();assert((await page.locator('h1').textContent()).includes('SEM-001'),'source_snapshot');await page.locator('a').first().click();assert(await page.locator('h1').textContent()==='从三维重建，到语义理解','source_return');
 await page.waitForFunction(()=>[...document.images].filter(x=>x.getAttribute('src')&&x.loading!=='lazy').every(x=>x.complete&&x.naturalWidth>0));
 assert(network.length===0,'zero_network_requests');assert(errors.length===0,'zero_browser_errors');
 fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify({status:'passed',checks,errors,network},null,2));
 console.log(JSON.stringify({status:'passed',checks:Object.keys(checks).length,errors,network}));
}catch(e){await page.screenshot({path:path.join(qa,'failure.png'),fullPage:true});fs.writeFileSync(path.join(qa,'failure.json'),JSON.stringify({error:String(e),checks,network,errors},null,2));throw e;}finally{await browser.close()}
