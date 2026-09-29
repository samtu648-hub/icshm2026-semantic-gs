#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const opt=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,r)=>i%2?a:[...a,[v.replace(/^--/,''),r[i+1]]],[]));
const require=createRequire(import.meta.url);const {chromium}=require(path.join(opt['node-modules'],'playwright'));
const report=path.resolve(opt.report),qa=path.resolve(opt.qa);fs.mkdirSync(qa,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:opt.chrome||'/usr/bin/google-chrome',args:['--no-sandbox','--disable-gpu']});
const context=await browser.newContext({viewport:{width:1440,height:900},offline:true});
const page=await context.newPage();const errors=[],network=[],checks={};page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
const assert=(condition,label)=>{if(!condition)throw Error(label);checks[label]=true;};
try{
 await page.goto(pathToFileURL(path.join(report,'index.html')).href);await page.waitForFunction(()=>document.querySelectorAll('.compare-grid canvas').length>10);await page.waitForTimeout(600);
 assert(await page.locator('.slide').count()===10,'ten_slides');
 for(const size of [{width:1440,height:900},{width:1366,height:768},{width:1920,height:1080},{width:390,height:844}]){
  await page.setViewportSize(size);
  for(let i=1;i<=10;i++){
   await page.evaluate(n=>navigate(n-1),i);await page.waitForTimeout(50);
   assert(await page.locator('#slide-'+i).isVisible(),`visible_${i}_${size.width}`);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`no_horizontal_overflow_${i}_${size.width}`);
   if(size.width===1440||size.width===1366)await page.screenshot({path:path.join(qa,`${size.width}-slide-${i}.png`),fullPage:true});
  }
 }
 await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>navigate(0));await page.locator('h1').click();await page.keyboard.press('ArrowRight');assert(await page.locator('#slide-2').isVisible(),'keyboard_next');await page.keyboard.press('ArrowLeft');assert(await page.locator('#slide-1').isVisible(),'keyboard_previous');
 await page.locator('#toc').click();await page.locator('#toc-list button').nth(6).click();assert(await page.locator('#slide-7').isVisible(),'toc_jump');
 const v=page.locator('#slide-7 video').first();await v.evaluate(v=>{v.load();});await v.evaluate(async v=>{await v.play();});await page.waitForTimeout(500);assert(await v.evaluate(v=>v.currentTime>0&&!v.paused),'video_plays_offline');await page.locator('[data-seek="72"]').click();await page.waitForTimeout(150);assert(await v.evaluate(v=>Math.abs(v.currentTime-6)<.15&&v.paused),'video_frame_seek');
 await page.locator('#slide-7 .video-controls select').first().selectOption('0.5');assert(await v.evaluate(v=>v.playbackRate===.5),'video_speed');
 await page.locator('#slide-7 summary').click();const cmp=page.locator('#slide-7 .comparison');await cmp.locator('[data-box="box"]').click();assert(await cmp.locator('canvas').first().evaluate(c=>c.width===700&&c.height===369),'same_coordinate_crop');
 await cmp.locator('select').selectOption('160.png');await page.waitForTimeout(150);assert(await cmp.locator('canvas').first().evaluate(c=>c.width===1320),'view_switch');await cmp.locator('[data-box="full"]').click();const canvas=cmp.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();await page.mouse.move(box.x+box.width*.2,box.y+box.height*.2);await page.mouse.down();await page.mouse.move(box.x+box.width*.7,box.y+box.height*.7);await page.mouse.up();assert(await cmp.locator('canvas').evaluateAll(cs=>cs.every(c=>c.width===cs[0].width&&c.width<1320)),'synchronized_drag_zoom');
 await cmp.locator('input[type=checkbox]').first().check();assert(await cmp.locator('canvas').count()===3,'method_toggle');
 await page.locator('#appendix-btn').click();await page.locator('#search').fill('SEM-376');assert(await page.locator('#records a').count()===1,'registry_search');await page.locator('#records a').click();assert(await page.locator('h1').textContent()==='SEM-376-CROSS-RGB-INFERENCE','source_snapshot_opens');await page.locator('a').first().click();assert(await page.locator('#appendix').isVisible(),'source_return');await page.locator('#return-slide').click();assert(await page.locator('.slide:visible').count()===1,'appendix_return');
 await page.locator('#fullscreen').click();assert(await page.evaluate(()=>!!document.fullscreenElement),'fullscreen');await page.locator('#fullscreen').click();
 await page.evaluate(()=>appendix('extra-videos'));for(const key of ['intro','backbone','matched','cross','joint','depth']){const media=page.locator(`[data-video="${key}"] video`).first();await media.evaluate(async v=>{v.load();await v.play();});await page.waitForTimeout(200);assert(await media.evaluate(v=>v.currentTime>0&&v.videoWidth>0),`decoded_${key}`);await media.evaluate(v=>v.pause());}
 assert(await page.locator('[data-load-error]').count()===0,'all_comparison_images_load');assert(network.length===0,'zero_network_requests');assert(errors.length===0,'zero_browser_errors');
 fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify({status:'passed',report,checks,network,errors},null,2));console.log(JSON.stringify({status:'passed',checks:Object.keys(checks).length,network,errors}));
}catch(e){await page.screenshot({path:path.join(qa,'failure.png'),fullPage:true});fs.writeFileSync(path.join(qa,'failure.json'),JSON.stringify({error:String(e),checks,errors,network},null,2));throw e;}finally{await browser.close();}
