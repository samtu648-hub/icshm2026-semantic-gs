#!/usr/bin/env node
// Real file:// offline browser checks for REPORT-003.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const options=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,r)=>i%2?a:[...a,[v.replace(/^--/,''),r[i+1]]],[]));
const require=createRequire(import.meta.url);
const {chromium}=require(path.join(options['node-modules'],'playwright'));
const report=path.resolve(options.report),qa=path.resolve(options.qa);
fs.mkdirSync(qa,{recursive:true});
const browser=await chromium.launch({executablePath:options.chrome||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']});
const context=await browser.newContext({offline:true,viewport:{width:1440,height:900}});
const page=await context.newPage();const errors=[],network=[],checks={};
page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
const assert=(condition,label)=>{if(!condition)throw Error(label);checks[label]=true};
try{
 await page.goto(pathToFileURL(path.join(report,'index.html')).href);
 await page.waitForFunction(()=>document.images.length>0&&[...document.images].every(x=>x.complete&&x.naturalWidth>0));
 assert(await page.locator('main>section').count()===9,'nine_sections');
 assert((await page.locator('h1').textContent()).includes('可信的停止点'),'opening_message');
 for(const size of [{width:1440,height:900},{width:1366,height:768},{width:390,height:844}]){
  await page.setViewportSize(size);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`no_overflow_${size.width}`);
  for(const id of ['opening','journey','horizon','hard','best']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();
   if(size.width!==1366)await page.locator('#'+id).screenshot({path:path.join(qa,`${size.width}-${id}.png`)});
  }
 }
 await page.setViewportSize({width:1440,height:900});
 await page.locator('#journey').scrollIntoViewIfNeeded();
 for(let i=0;i<4;i++){
  await page.locator('.stage').nth(i).click();const v=page.locator('#journey-video');
  await v.evaluate(async x=>{await x.play()});await page.waitForTimeout(250);
  assert(await v.evaluate(x=>x.videoWidth>0&&x.videoHeight>0&&x.currentTime>0&&!x.paused),`journey_video_${i}_decodes`);
  await v.evaluate(x=>x.pause());
 }
 for(const view of ['049','057','146','159']){
  await page.locator(`[data-view="${view}"]`).click();
  await page.waitForFunction(()=>[...document.querySelectorAll('#example-grid img')].every(x=>x.complete&&x.naturalWidth>0));
  assert(await page.locator('#example-grid img').count()===4,`example_${view}_four_tiles`);
 }
 await page.locator('#best').scrollIntoViewIfNeeded();
 for(let i=0;i<4;i++){
  const v=page.locator('#best video').nth(i);await v.evaluate(async x=>{await x.play()});await page.waitForTimeout(180);
  assert(await v.evaluate(x=>[360,720].includes(x.videoWidth)&&x.videoHeight===270&&x.currentTime>0),`best_video_${i}_decodes`);await v.evaluate(x=>x.pause());
 }
 assert(network.length===0,'zero_network_requests');assert(errors.length===0,'zero_browser_errors');
 fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify({status:'passed',checks,errors,network},null,2));
 console.log(JSON.stringify({status:'passed',checks:Object.keys(checks).length,errors,network}));
}catch(e){await page.screenshot({path:path.join(qa,'failure.png'),fullPage:true});fs.writeFileSync(path.join(qa,'failure.json'),JSON.stringify({error:String(e),checks,network,errors},null,2));throw e;}finally{await browser.close()}
