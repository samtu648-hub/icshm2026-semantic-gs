#!/usr/bin/env node
// Anonymous online browser acceptance for the published REPORT005 repository and Pages site.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const args=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,r)=>i%2?a:[...a,[v.replace(/^--/,''),r[i+1]]],[]));
const require=createRequire(import.meta.url);
const {chromium}=require(path.join(args['node-modules'],'playwright'));
const report=args.url.replace(/\/$/,'')+'/',qa=path.resolve(args.qa);
fs.mkdirSync(qa,{recursive:true});
const browser=await chromium.launch({executablePath:args.chrome||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required']});
const context=await browser.newContext({viewport:{width:1440,height:900}});
context.setDefaultTimeout(60000); context.setDefaultNavigationTimeout(90000);
const page=await context.newPage();const errors=[],network=[],checks=[],httpFailures=[]; page.on('response',r=>{if(r.url().startsWith(report)&&r.status()>=400)httpFailures.push({url:r.url(),status:r.status()})});
page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
const assert=(ok,label)=>{if(!ok)throw Error(label);checks.push(label)};
try{
 await page.goto(new URL('index.html',report).href);
 assert(await page.locator('#experiment-table tbody tr').count()===351,'351 experiment rows');
 assert(await page.locator('.figure').count()===10,'ten figure themes');
 assert(await page.locator('video').count()===8,'eight playable videos');
 await page.locator('.downloads a[href="author_guide.html"]').click();
 await page.waitForLoadState('domcontentloaded');
 await page.waitForFunction(()=>document.querySelectorAll('.guide-glossary dt').length===16);
 assert(await page.locator('.guide-chapter').count()===12,'twelve readable author chapters');
 assert(await page.locator('.guide-glossary dt').count()===16,'sixteen glossary entries');
 for(const size of [{width:1440,height:900},{width:390,height:844}]){
  await page.setViewportSize(size);
  for(const id of ['contents','A04','A09','glossary']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();
   await page.locator('#'+id+' img').evaluateAll(images=>images.forEach(x=>x.loading='eager'));
   await page.waitForFunction(id=>[...document.querySelectorAll('#'+id+' img')].every(x=>x.complete&&x.naturalWidth>0),id);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`author_no_overflow_${size.width}_${id}`);
   await page.screenshot({path:path.join(qa,`${size.width}-guide-${id}.png`)});
  }
 }
 await page.locator('#A04 .guide-check summary').click();
 assert(await page.locator('#A04 .guide-check p').isVisible(),'author self-check answer expands');
 assert((await page.locator('#A04 .guide-check p').textContent()).includes('三条不同阶段'),'author answer explains mixed-system mistake');
 await page.locator('#A09 .guide-media a').first().click();
 assert(page.url().includes('index.html#F08'),'author evidence link reaches figure library');
 await page.goto(new URL('index.html',report).href);
 for(const size of [{width:1440,height:900},{width:1366,height:768},{width:390,height:844}]){
  await page.setViewportSize(size);
  await page.locator('#start').scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(qa,`${size.width}-opening.png`)});
  for(const id of ['F03','F07','F09','F10','experiments']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();
   await page.locator('#'+id+' img').evaluateAll(images=>images.forEach(x=>x.loading='eager'));
   await page.waitForFunction(id=>[...document.querySelectorAll('#'+id+' img')].every(x=>x.complete&&x.naturalWidth>0),id);
   const fits=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1);
   if(!fits)fs.writeFileSync(path.join(qa,'overflow.json'),JSON.stringify(await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,elements:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&!e.closest('.table-wrap')).slice(0,20).map(e=>({tag:e.tagName,class:e.className,width:e.getBoundingClientRect().width,text:e.textContent.slice(0,90)}))})),null,2));
   assert(fits,`no_overflow_${size.width}_${id}`);
   if(id!=='experiments'&&size.width!==1366)await page.locator('#'+id).screenshot({path:path.join(qa,`${size.width}-${id}.png`)});
  }
 }
 await page.setViewportSize({width:1440,height:900});
 await page.locator('#search').fill('SEM-161');
 assert(await page.locator('#experiment-table tbody tr:visible').count()>=1,'search resolves experiment');
 await page.locator('#search').fill('no-such-record-xyz');
 assert(await page.locator('#empty').isVisible(),'empty search response');
 await page.locator('#search').fill('');
 await page.locator('#state').selectOption('阻塞');
 assert(await page.locator('#experiment-table tbody tr:visible').count()>0,'status filter');
 await page.locator('#state').selectOption('');
 assert(await page.locator('#experiment-table tbody tr:visible').count()===351,'filter reset');
 await page.locator('#F03 summary').click();
 await page.waitForFunction(()=>[...document.querySelectorAll('#F03 img')].every(x=>x.complete&&x.naturalWidth>0));
 assert(await page.locator('#F03 .extra-images img').isVisible(),'paired example expanded');
 const docs=await page.locator('#F03 .source a').first().getAttribute('href');
 await page.goto(new URL(docs,report).href);
 assert((await page.locator('body').textContent()).includes('git_commit'),'portable report and provenance detail');
 await page.goto(new URL('index.html',report).href);
 for(let i=0;i<8;i++){
  const v=page.locator('video').nth(i);await v.scrollIntoViewIfNeeded();
  await v.evaluate(async x=>{await x.play()});
  await page.waitForFunction(i=>{const x=document.querySelectorAll('video')[i];return x.videoWidth>0&&x.currentTime>0&&!x.paused},i);
  assert(await v.evaluate(x=>x.videoWidth>0&&x.currentTime>0&&!x.paused),`video_${i}_plays`);
  await v.evaluate(x=>{x.pause();x.currentTime=x.duration*.6});
  await page.waitForFunction(i=>{const x=document.querySelectorAll('video')[i];return !x.seeking&&x.readyState>=2&&x.currentTime>x.duration*.5},i);
  assert(await v.evaluate(x=>x.currentTime>x.duration*.5),`video_${i}_seeks`);
 }
 // 200% text enlargement still keeps the document within the viewport.
 await page.addStyleTag({content:'body{font-size:34px}'});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'enlarged text fits');
 assert(httpFailures.length===0,'zero failed site responses');assert(errors.length===0,'zero browser errors');
 const github=await context.newPage();
 const repo='https://github.com/samtu648-hub/icshm2026-semantic-gs';
 for(const suffix of ['', '/blob/main/docs/paper/author_guide.md','/blob/main/docs/paper/team_comparison.md','/releases/tag/paper-handoff-20260929-v1']){
  const response=await github.goto(repo+suffix,{waitUntil:'domcontentloaded'});
  assert(response.status()===200,'anonymous GitHub '+suffix);
  await github.locator('main').waitFor();
 }
 await github.goto(repo,{waitUntil:'domcontentloaded'});
 await github.locator('article').waitFor();
 assert((await github.locator('article').innerText()).includes('当前模型身份'),'anonymous rendered README');
 assert((await context.cookies()).every(c=>c.name!=='user_session'),'browser has no authenticated GitHub session');
 await github.screenshot({path:path.join(qa,'github-readme.png'),fullPage:true});
 const result={status:'passed',checks:checks.length,labels:checks,errors,network};
 fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(e){await page.screenshot({path:path.join(qa,'failure.png')});fs.writeFileSync(path.join(qa,'failure.json'),JSON.stringify({error:String(e),checks,network,errors},null,2));throw e}finally{await browser.close()}
