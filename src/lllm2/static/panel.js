const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const settingKeys=['model','engine','backend','device','context','slots','gpu_layers','flash','cache','cache_k','cache_v','speculation','drafter','pair_confirmed','draft_length','effort','draft_cache','chat_template','batch_size','ubatch_size','backend_sampling','cuda_graph_opt','cache_ram_mib','context_checkpoints','lookup_ngram_n','lookup_ngram_m'];
const optionalNumbers=['gpu_layers','batch_size','ubatch_size','cache_ram_mib','context_checkpoints','lookup_ngram_n','lookup_ngram_m'];
const optionalCache=['cache_k','cache_v'];
const cachePair=s=>[s.cache_k||s.cache,s.cache_v||s.cache];
const cacheLabel=s=>{const [k,v]=cachePair(s);return `K ${k} / V ${v}`;};
const settingValue=(s,k)=>optionalNumbers.includes(k)||optionalCache.includes(k)?s[k]??null:k==='backend_sampling'?s[k]??false:k==='cuda_graph_opt'?s[k]??'default':s[k];
// Shared launch and feature copy; help buttons stay outside labels.
const launchHelp={
 model:['Installed checkpoint','The GGUF file contains the model weights. Size and precision affect memory, speed and answers. Keep the recommended checkpoint unless comparing models.'],
 engine:['llama-server binary','The program that runs the checkpoint. Builds support different devices and features; changing it can affect speed and compatibility. Keep a working build for fair comparisons.'],
 backend:['Backend','CUDA and Vulkan are ways to run the model on your NVIDIA GPU. Support and speed can differ; keep the recommended backend unless comparing them.'],
 device:['GPU device','The GPU reported by this engine for the chosen backend. Its free memory limits what fits. Keep the detected device unless deliberately using another GPU.'],
 context:['Total allocated context','Text capacity in tokens, including prompts and replies, shared across slots. More needs more memory and can take longer to fill. Keep the recommended allocation unless you need a different context.'],
 slots:['Slots','How many requests can hold their own conversation state. Total context is divided between slots; more slots leave less per request. This is not CPU threads. Keep the recommended value for normal use.'],
 batch_size:['Logical batch (tokens)','Maximum prompt tokens submitted together. A larger batch may read prompts faster but uses more working memory. Leave blank for this engine’s default; microbatch must not exceed it.'],
 backend_sampling:['Target GPU sampling','Experimental: choose the next target token on the GPU. This may reduce overhead, but unsupported sampler requests can fall back. Leave unchecked for the engine default until measured. Draft sampling is unchanged.'],
 cuda_graph_opt:['Concurrent CUDA streams','Experimental: overlap eligible operations inside a CUDA graph. Requires one visible CUDA device and ordinary CUDA Graphs. Default preserves the engine environment; this does not add concurrent requests.'],
 cache_ram_mib:['Host prompt cache (MiB)','Finite RAM allowance for saved conversations. Zero disables saved conversation caching. Blank keeps the normal engine default; the warm experiment uses 2048 MiB when blank. This is not a total process RAM cap.'],
 context_checkpoints:['Context checkpoints per slot','Saved states can help resume edited history on recurrent models. Zero disables these checkpoints. Blank keeps the normal engine default; the warm experiment uses four when blank. More checkpoints use additional memory.'],
 ubatch_size:['Physical microbatch (tokens)','Prompt tokens processed together in one physical chunk of the logical batch. Larger chunks may improve prefill speed but need more GPU memory. Leave blank for the engine default until compared.'],
 gpu_layers:['GPU layers','Blank means automatic: the engine fits model weights and buffers to free GPU memory, keeping a 1 GiB margin and using system RAM when needed. Requires an engine with memory fitting support. Enter a number to override; 999 requests all layers and 0 keeps model layers on the CPU. Actual placement appears in the engine log.'],
 flash:['Flash attention','Calculates attention with less working memory and may improve speed. Auto leaves the choice to the engine. Keep the recommended value; quantized KV cache and DFlash require on here.'],
 cache:['Common conversation memory precision','Explicitly selecting this links K and V and clears Advanced overrides. The actual pair is shown below. Stores information about text already read. q8_0 and q4_0 use less memory than f16; speed and accuracy can change. Keep the starting precision until compared.'],
 effort:['Reasoning effort','Asks a compatible model/template to spend more or less effort reasoning. More can take longer; accepted values depend on the template. Keep default unless testing a supported value.'],
 lookup_ngram_n:['Lookup match N','Number of matching tokens used by ngram-simple. Set N and M together with 1 ≤ N ≤ M; N greater than M produces no drafts on the verified build. Blank preserves old modes and selects N3 for MTP+lookup.'],
 lookup_ngram_m:['Lookup draft M','Lookup continuation width, independent of MTP draft length. Set together with N. Blank preserves old modes and selects M3 for MTP+lookup. Compare equal widths first; extra guesses can add rejected work.'],
 'draft-mtp,ngram-simple':['MTP + lookup · experimental','Explicit combination requiring both engine methods, lookup controls and checkpoint MTP tensors. Build 662a0b0 tries lookup before MTP fallback. Ordinary HTTP counters aggregate both methods; no speedup is implied.'],
 speculation:['Guess ahead (speculative decoding)','Checks guessed tokens together to try to generate answers faster. Choose MTP, DFlash, prompt lookup or the explicit MTP+lookup experiment; overhead can outweigh savings. Keep the starting mode until measured.'],
 'draft-mtp':['MTP','Uses a prediction head inside the checkpoint to guess upcoming tokens. It needs MTP tensors and engine support. Longer guesses can add wasted work; keep the starting draft length.'],
 'draft-dflash':['DFlash · experimental','Uses a separate target-specific model to guess tokens. Extra memory and checking work can outweigh savings. Requires CUDA, flash attention on and a verified pair; use only for a deliberate comparison.'],
 'ngram-simple':['Prompt lookup (ngram)','Guesses continuations from repeated text, without a draft model. It may help copying code but adds checking work when guesses miss. Leave the starting mode unless testing repetitive tasks.'],
 draft_length:['Draft length','Maximum tokens guessed at once when the engine supports this control. Longer guesses may help, or waste more work when rejected. Keep the starting length; unused with speculation off.'],
 cache_k:['K cache override','Inherit uses common precision; explicit values override only attention K. Mixed pairs require engine/backend verification. Draft cache is unchanged.'],
 cache_v:['V cache override','Inherit uses common precision; explicit values override only attention V. Quantized cache requires flash attention on here. Mixed types may lack a GPU kernel.'],
 cache_pair:['Actual cache pair','Resolved K and V include Advanced overrides. Compiled kernel evidence is separate from runtime execution and measured capacity.'],
 draft_cache:['Draft KV cache','Conversation-memory precision for MTP or DFlash guesses. Lower precision may save memory but needs engine support and can affect speed. Default leaves it to the engine; keep the starting value.'],
 chat_template:['Chat template override (optional)','A local template file formats messages and reasoning instructions. A mismatch can cause errors or change answers. Keep the supplied template; blank uses the checkpoint template.'],
 drafter:['DFlash drafter GGUF','Local path to the extra model used only by DFlash. It must match this exact target and uses extra memory. Browse selects a file; it does not prove compatibility. Leave alone unless testing DFlash.'],
 pair_confirmed:['Verify the DFlash pair','Confirm training and conversion compatibility for this exact target. A matching filename is not evidence; a wrong pair can fail or behave incorrectly. Leave unchecked unless verified.']
};
function helpTip(key,prefix){const [name,description]=launchHelp[key]||[key,'Check engine and model compatibility before comparing performance.'];const id=`${prefix}-tip-${key}`;return `<span class="tip"><button type="button" aria-label="${esc(name)} help" aria-describedby="${id}" aria-expanded="false">ⓘ</button><span class="tip-content" role="tooltip" id="${id}">${esc(description)}</span></span>`;}
function launchTips(){
 for(const key of settingKeys){
  const control=$(key), label=control.closest('label')||document.querySelector(`label[for="${key}"]`);
  if(control.type==='checkbox'){const row=document.createElement('div');row.className='row wide';label.replaceWith(row);row.append(label);row.insertAdjacentHTML('beforeend',helpTip(key,'launch'));continue;}
  label.htmlFor=key;
  if(label.contains(control)){const field=document.createElement('div');field.className='field-control'+(label.classList.contains('wide')?' wide':'');label.replaceWith(field);const title=document.createElement('div');title.className='field-title';field.append(title);title.append(label);field.append(control);for(const child of [...label.children])field.append(child);label.textContent=launchHelp[key][0];title.insertAdjacentHTML('beforeend',helpTip(key,'launch'));}
  else{label.textContent=launchHelp[key][0];label.insertAdjacentHTML('afterend',helpTip(key,'launch'));}
 }
}
let featureState=null;
function renderFeatures(){
 if(!featureState)return;
 const s=settings(), statuses={experimental:'Experimental · runtime unverified',available:'Available to try',unsupported:'Not supported',unknown:'Support unconfirmed','missing prerequisites':'Needs setup'};
 $('features').innerHTML=Object.entries(featureState).map(([key,v])=>{
  const choice=key in s?`Form value: ${s[key]??'engine default'}`:s.speculation===key?'Selected in form':'Not selected in form';
  const reason=key==='flash'||key==='cache'||key==='effort'?(v.status==='unsupported'?'This engine does not advertise the required control.':v.status==='unknown'?'Support could not be confirmed for this engine and template.':'Engine control detected; this model/backend still needs a successful launch.'):v.reason;
  return `<div class="feature"><div class="field-title"><b>${esc((launchHelp[key]||[key])[0])}</b>${helpTip(key,'feature')}</div><div>${esc(statuses[v.status]||'Support unconfirmed')} · ${esc(choice)}</div><small>${esc(reason)}</small><details><summary>Technical diagnostics</summary><small>${esc(key)} · ${esc(v.status)}</small><small>${esc(v.reason)}</small></details></div>`;
 }).join('');
 initTips($('features'));
}
let token='', discovered={}, combinations=[], results=[], lastResults='', statusState={}, loadedDefaults=null;
let view='launch', drafts={launch:null,experiments:null}, connected=false, resolving=true, selectionSequence=0, validationSequence=0;
let validationSnapshot='', validationError='', actionError='', selectionNote='', pendingAction=false, pollPending=false, scanPending=false;
let savedExists=false, engineOverride=false, editTimer=null, lastDownloads='', startAttempt=null, discoveredOnce=false, queuedView=null;
let savedFeedback='', pickerSequence=0;
const sameSettings=(a,b)=>!!a&&!!b&&settingKeys.every(k=>settingValue(a,k)===settingValue(b,k));
const snapshot=()=>JSON.stringify(settings());
const currentLaunch=()=>view==='launch'?settings():drafts.launch?.settings;
const modelName=path=>{const m=discovered.models?.find(m=>m.path===path),c=discovered.catalog?.find(c=>c.id===m?.catalog_id);return c?(c.recommendation?.name||c.name):path?.split('/').slice(-2).join('/')||'No model selected';};

function settings(){return Object.fromEntries(settingKeys.map(k=>[k,$(k).type==='checkbox'?$(k).checked:$(k).type==='number'?(optionalNumbers.includes(k)&&$(k).value===''?null:Number($(k).value)):optionalCache.includes(k)?$(k).value||null:$(k).value]));}
function slotNote(){$('slot-note').textContent=`${Math.floor(Number($('context').value)/Number($('slots').value)).toLocaleString()} tokens per conversation (slot), including the reply. Total allocation is divided across ${$('slots').value} slot(s).`;benchmarkCost();}
function message(s,error=false){$('message').textContent=s;$('message').style.display='block';$('message').className=error?'error':'';}
async function api(path,data){const r=await fetch(path,{signal:AbortSignal.timeout(path==='/api/start'?15000:path==='/api/status'?10000:180000),...(data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-LLLM2-Token':token},body:JSON.stringify(data)})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d;}
async function attempt(fn){try{await fn();}catch(e){message(e.message,true);}}
let resultSort={key:'started',direction:'descending'}, resultRequest=0;
let resultDeletion=null,resultDeleteBusy=false;
const deletableResult=r=>['complete','failed','cancelled','interrupted'].includes(r.status);
const failedResult=r=>['failed','cancelled'].includes(r.status);
const expandedResults=new Set();
const resultFormat=value=>Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:1}):'—';
const sampleMode=(r,s)=>r.measurement_mode==='warm-conversation'?(s?.control?'Uncached replay':'Warm conversation'):'Cold benchmark';
const sampleStatus=(r,s)=>s?.status||r.status||'unknown';
function statusClass(status){return status==='complete'?'complete':['failed','error'].includes(status)?'failed':['cancelled','interrupted','partial'].includes(status)?'partial':'';}
function statusBadge(status){return `<span class="result-status ${statusClass(status)}">${esc(status)}</span>`;}
function resultRows(){
 const rows=results.flatMap(r=>(r.samples.length?r.samples:[null]).map((s,i)=>({r,s,i,key:encodeURIComponent(r.id)+'-'+i})));
 const value=({r,s})=>({started:Date.parse(r.started),status:sampleStatus(r,s),prefill:r.measurement_mode==='warm-conversation'?s?.processed_prefill_tok_s:s?.prefill_tok_s,decode:s?.decode_tok_s,input:s?.input_tokens,gpu:s?.peak_total_gpu_used_mib})[resultSort.key];
 const missing=v=>v==null||(typeof v==='number'&&!Number.isFinite(v));
 return rows.sort((a,b)=>{
  const av=value(a),bv=value(b);if(missing(av)||missing(bv))return Number(missing(av))-Number(missing(bv));
  const comparison=typeof av==='string'?av.localeCompare(bv):av-bv;
  return resultSort.direction==='ascending'?comparison:-comparison;
 });
}
function resultDetails({r,s,i,key}){
 const fmt=resultFormat,warm=r.measurement_mode==='warm-conversation',allocated=r.settings.context,slots=r.settings.slots;
 const {samples,...record}=r;
 return `<div class="row result-actions">${!resultBlock(r)?`<button id="promote-${key}" data-promote="${esc(r.id)}" data-has-context="${!!r.largest_observed_context}">Try in Launch</button>${r.largest_observed_context?`<span class="context-load-actions" role="radiogroup" aria-label="Context to use in Launch">${headroomControl(r,key)}</span>`:''}`:''}<span class="spacer"></span><button data-copy-row="${key}" id="copy-row-${key}">Copy row</button><button id="delete-run-${key}" data-delete-result="${esc(r.id)}" aria-label="Delete entire run: ${esc(r.label)}" ${deletableResult(r)?'':'disabled'}>Delete run…</button></div>
 <div class="result-detail-grid"><div><b>Configuration</b><p>${esc(r.settings.model)}</p><p>${esc(r.settings.backend)} · ${esc(r.settings.device)} · ${esc(r.settings.speculation)} · ${esc(cacheLabel(r.settings))}</p><p>${fmt(allocated/slots)} tokens per conversation · ${fmt(allocated)} total / ${fmt(slots)} slots</p></div>
 <div><b>Sample & timing</b><p>${esc(sampleMode(r,s))}${s?.turn?' · '+esc(s.turn):''} · ${esc(s?.workload||'No sample')}</p><p>${fmt(s?.input_tokens)} input / ${fmt(s?.output_tokens)} output tokens · ${fmt(s?.wall_seconds??s?.completion_seconds)} s elapsed</p><p>Output requested / cap: ${fmt(s?.requested_output_budget)} / ${fmt(s?.output_budget)}</p>${warm?`<p>Processed ${fmt(s?.processed_tokens)} · reused ${fmt(s?.reused_tokens)} tokens</p><p>First token event ${fmt(s?.first_token_event_seconds)} s · first text ${fmt(s?.first_text_seconds)} s · completion ${fmt(s?.completion_seconds)} s</p>`:''}<p>Peak engine RSS ${fmt(s?.peak_engine_rss_mib)} MiB · sampled total VRAM ${fmt(s?.peak_total_gpu_used_mib)} MiB</p>${s?.adherence?`<p>Source adherence: ${esc(s.adherence.status)}</p>`:''}</div>
 <div><b>Context search</b><p>Observed ${fmt(r.largest_observed_context)} / headroom estimate ${fmt(r.recommended_context)} tokens per conversation${r.context_confirmed===false?' (loaded only; not confirmed with a full prompt)':''}</p><p>${esc(r.context_search_status||'No context search recorded')}${r.context_failed_upper_bound?' · failed upper bound '+fmt(r.context_failed_upper_bound):''}${r.context_timeout_upper_bound?' · timed out at '+fmt(r.context_timeout_upper_bound):''}${r.largest_started_context&&r.largest_started_context!==r.largest_observed_context?' · loaded up to '+fmt(r.largest_started_context)+' (unconfirmed)':''}</p><p>${esc(r.context_search_stop_reason||'')}${r.context_ceiling_reached?' Search ceiling reached; maximum may be higher.':''}</p></div>
 <div><b>Run status</b><p>${statusBadge(r.status)} · ${esc(r.started)}</p>${r.quality_status?`<p>Quality/adherence: ${esc(r.quality_status)}</p>`:''}<p class="error">${esc([s?.error,r.error].filter(Boolean).join('\n'))}</p><p>Result: ${esc(r.id)} · sample ${s?i+1:'unavailable'}</p></div></div>
 <details id="result-evidence-${key}"><summary id="result-evidence-summary-${key}">Exact settings & measurement evidence</summary><pre>${esc(JSON.stringify({...record,sample:s},null,2))}</pre></details>`;
}
function renderResults(){
 const rows=resultRows(),fmt=resultFormat;
 $('export-csv').disabled=$('copy-results').disabled=!rows.length;
 const samples=rows.filter(row=>row.s).length,status=rows.length?`${samples} sample${samples===1?'':'s'} · ${results.length} run${results.length===1?'':'s'} · sorted by ${({started:'run date',status:'sample status',prefill:'prefill speed',decode:'decode speed',input:'input tokens',gpu:'peak VRAM'})[resultSort.key]}, ${resultSort.direction}.`:'No experiments yet.';
 if($('results-status').textContent!==status)$('results-status').textContent=status;
 document.querySelectorAll('[data-sort]').forEach(b=>{const direction=b.dataset.sort===resultSort.key?resultSort.direction:'none';b.closest('th').setAttribute('aria-sort',direction);b.querySelector('span').textContent=direction==='ascending'?' ↑':direction==='descending'?' ↓':' ↕';});
 const liveKeys=new Set(rows.map(row=>row.key));for(const key of expandedResults)if(!liveKeys.has(key))expandedResults.delete(key);
 renderMarkup('results',rows.map(row=>{
  const {r,s,key}=row,warm=r.measurement_mode==='warm-conversation',status=sampleStatus(r,s),expanded=expandedResults.has(key);
  return `<tr data-result-key="${key}"><td class="result-name"><div class="result-head"><button class="result-expander" id="expand-${key}" data-expand="${key}" aria-expanded="${expanded}" aria-controls="detail-${key}" aria-label="${expanded?'Hide':'Show'} details for ${esc(r.label)}, sample ${row.i+1}">${expanded?'▾':'▸'}</button><div><b>${esc(r.label)}</b><small>${esc(s?.workload||'No sample')} · ${esc(sampleMode(r,s))}${s?.turn?' · '+esc(s.turn):''}</small><small class="result-model" title="${esc(r.settings.model)}">${esc(modelName(r.settings.model))} · ${esc(r.settings.backend)} · ${fmt(Math.floor(r.settings.context/r.settings.slots))} context</small><small>${esc(r.started)}</small></div></div></td>
  <td>${statusBadge(status)}${status!==r.status?`<small>Run: ${esc(r.status)}</small>`:''}${r.quality_status==='failed'||s?.adherence?.status==='failed'?'<small class="error">Adherence failed</small>':''}</td>
  <td>${fmt(warm?s?.processed_prefill_tok_s:s?.prefill_tok_s)}${warm?'<small>Processed tokens only</small>':''}</td><td>${fmt(s?.decode_tok_s)}</td><td>${fmt(s?.input_tokens)} / ${fmt(s?.output_tokens)}</td><td>${fmt(s?.peak_total_gpu_used_mib)}</td></tr>
  <tr class="result-detail" id="detail-${key}" ${expanded?'':'hidden'}><td colspan="6">${resultDetails(row)}</td></tr>`;
 }).join('')||'<tr><td colspan="6">No experiments yet.</td></tr>');
 resultDeleteControls();
}
function resultDeleteControls(){
 const blocked=!connected||!!statusState.job?.active||resultDeleteBusy;
 const failed=results.filter(failedResult).length;
 $('delete-failed-results').disabled=blocked||!failed;
 $('delete-failed-results').textContent=`Delete failed / cancelled${failed?' ('+failed+')':''}…`;
 for(const button of document.querySelectorAll('[data-delete-result]')){
  const run=results.find(r=>r.id===button.dataset.deleteResult);
  button.disabled=blocked||!run||!deletableResult(run);
 }
}
function openResultDeletion(runs,failedOnly){
 if(!runs.length||resultDeleteBusy)return;
 resultDeletion={result_ids:runs.map(r=>r.id),failed_only:failedOnly};
 const samples=runs.reduce((n,r)=>n+r.samples.length,0);
 $('delete-results-description').textContent=runs.length===1?`Delete “${runs[0].label}” (${runs[0].status}), with ${samples} sample${samples===1?'':'s'}?`:`Delete these ${runs.length} failed/cancelled runs, with ${samples} samples?`;
 $('delete-results-confirm').textContent=`Delete ${runs.length} run${runs.length===1?'':'s'}`;
 $('delete-results-error').textContent='';$('delete-results-dialog').showModal();$('delete-results-cancel').focus();
}
function forgetDeletedResultSources(ids){
 const forget=source=>source?.mode==='result'&&ids.includes(source.result_id)?{settings:source.settings,mode:'custom',source:'Custom settings · source experiment deleted',notes:['The source experiment was deleted. These settings can still be saved manually.']}:source;
 loadedDefaults=forget(loadedDefaults);
 for(const draft of Object.values(drafts))if(draft)draft.defaults=forget(draft.defaults);
 defaultState();
}
$('delete-failed-results').onclick=()=>openResultDeletion(results.filter(failedResult),true);
$('delete-results-cancel').onclick=()=>$('delete-results-dialog').close();
$('delete-results-dialog').addEventListener('cancel',e=>{if(resultDeleteBusy)e.preventDefault();});
$('delete-results-confirm').onclick=async()=>{
 if(resultDeleteBusy||!resultDeletion)return;
 resultDeleteBusy=true;resultDeleteControls();$('delete-results-confirm').disabled=$('delete-results-cancel').disabled=true;
 const selected=resultDeletion;
 try{
  const response=await api('/api/results/delete',selected);
  // Invalidate earlier GETs so they cannot resurrect rows after deletion.
  resultRequest++;const removed=new Set(selected.result_ids);
  results=results.filter(r=>!removed.has(r.id));lastResults=JSON.stringify(results);
  forgetDeletedResultSources(selected.result_ids);renderResults();
  $('delete-results-dialog').close();$('comparisons').focus();
  message(`Deleted ${response.deleted.length} experiment run${response.deleted.length===1?'':'s'}.`);
  await refreshResults();
 }catch(e){$('delete-results-error').textContent=e.message;}
 finally{resultDeleteBusy=false;$('delete-results-confirm').disabled=$('delete-results-cancel').disabled=false;resultDeleteControls();}
};
async function refreshResults(){
 const n=++resultRequest,data=await api('/api/results');if(n!==resultRequest)return;
 const key=JSON.stringify(data);if(key===lastResults){resultDeleteControls();return;}lastResults=key;results=data;renderResults();resultDeleteControls();
}
// Numeric rates and measurement modes remain separate in spreadsheet exports.
const resultColumns=[
 ['result_id',({r})=>r.id],['sample_number',({s,i})=>s?i+1:null],['started',({r})=>r.started],['label',({r})=>r.label],
 ['model',({r})=>r.settings.model],['engine',({r})=>r.settings.engine],['backend',({r})=>r.settings.backend],['device',({r})=>r.settings.device],
 ['model_sha256',({r})=>r.model?.sha256],['engine_sha256',({r})=>r.engine?.sha256],['hardware_json',({r})=>r.hardware?JSON.stringify(r.hardware):null],
 ['run_status',({r})=>r.status],['sample_status',({r,s})=>sampleStatus(r,s)],['measurement_mode',({r})=>r.measurement_mode||'cold'],['sample_kind',({r,s})=>sampleMode(r,s)],
 ['workload',({s})=>s?.workload],['turn',({s})=>s?.turn],['adherence_status',({s})=>s?.adherence?.status],['quality_status',({r})=>r.quality_status],
 ['input_tokens',({s})=>s?.input_tokens],['output_tokens',({s})=>s?.output_tokens],['requested_output_budget',({s})=>s?.requested_output_budget],['output_cap',({s})=>s?.output_budget],
 ['prefill_tok_s',({s})=>s?.prefill_tok_s],['processed_prefill_tok_s',({s})=>s?.processed_prefill_tok_s],['decode_tok_s',({s})=>s?.decode_tok_s],
 ['processed_tokens',({s})=>s?.processed_tokens],['reused_tokens',({s})=>s?.reused_tokens],['first_token_event_seconds',({s})=>s?.first_token_event_seconds],['first_text_seconds',({s})=>s?.first_text_seconds],['completion_seconds',({s})=>s?.completion_seconds],['wall_seconds',({s})=>s?.wall_seconds],
 ['peak_total_gpu_used_mib',({s})=>s?.peak_total_gpu_used_mib],['peak_engine_rss_mib',({s})=>s?.peak_engine_rss_mib],
 ['allocated_context_total',({r})=>r.settings.context],['slots',({r})=>r.settings.slots],['observed_context_per_slot',({r})=>r.largest_observed_context],['headroom_estimate_per_slot',({r})=>r.recommended_context],['context_search_status',({r})=>r.context_search_status],
 ['sample_error',({s})=>s?.error],['run_error',({r})=>r.error],['settings_json',({r})=>JSON.stringify(r.settings)]
];
function tableText(rows,separator=','){
 const cell=value=>{
  let text=value==null?'':String(value);
  // Prevent text labels/paths from becoming spreadsheet formulas; numeric data stays numeric.
  if(typeof value==='string'&&/^[\s]*[=+\-@]/.test(text))text="'"+text;
  return '"'+text.replace(/"/g,'""')+'"';
 };
 const newline=separator===','?'\r\n':'\n';
 return [resultColumns.map(([name])=>cell(name)).join(separator),...rows.map(row=>resultColumns.map(([,value])=>cell(value(row))).join(separator))].join(newline)+newline;
}
function downloadText(text,type,name){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export-csv').onclick=()=>downloadText('\ufeff'+tableText(resultRows()),'text/csv;charset=utf-8','lllm2-results.csv');
$('copy-results').onclick=()=>attempt(()=>copyText(tableText(resultRows(),'\t'),'Results table',false));
$('results-table').querySelector('thead').onclick=e=>{
 const b=e.target.closest('[data-sort]');if(!b)return;
 resultSort={key:b.dataset.sort,direction:resultSort.key===b.dataset.sort&&resultSort.direction==='descending'?'ascending':'descending'};renderResults();
};
function resetExperimentCeiling(modelPath=$('model').value){
 const model=discovered.models?.find(m=>m.path===modelPath);
 const entry=discovered.catalog?.find(m=>m.file===modelPath.split('/').pop());
 const ceiling=Math.min(model?.metadata.context||entry?.max_ctx||131072,1048576);
 $('max_context').value=ceiling;
 $('search-limit-note').textContent=`Search ceiling starts at ${ceiling.toLocaleString()} tokens per slot (${model?.metadata.context?'GGUF metadata':entry?.max_ctx?'catalogue limit':'generic fallback; model limit unknown'}). This is an upper bound to test, not a measured usable context. You can lower it.`;
}
function benchmarkCost(){
 const quick=$('sweep_prompts').checked, full=$('full_window').checked, search=$('search_context').checked;
 $('prompt_tokens').disabled=quick;
 $('max_context').disabled=$('context_timeout').disabled=!search;
 const hasSource=[...document.querySelectorAll('#workloads input:checked')].some(x=>['source-copy','source-small-edit','context-retrieval-edit'].includes(x.value));
 const upper=Math.floor(Number($('context').value)/Number($('slots').value))-Math.max(Number($('output_tokens').value),hasSource?2048:0)-32;
 const sizes=[...new Set([...(quick?[1024,16384,65536].map(n=>Math.min(n,upper)):[Number($('prompt_tokens').value)]),...(full?[upper]:[])])].sort((a,b)=>a-b);
 const workloads=document.querySelectorAll('#workloads input:checked').length, repeats=Number($('repeats').value);
 $('benchmark-time-tip').textContent='Rough guide from an observed Qwen/Vulkan run; other models, engines and hardware may take longer or finish sooner. Smaller windows may finish sooner. Suites and selected combinations repeat the work for each configuration. Context probes are additional and can take several minutes each. This is not a time limit.';
 if(!Number.isFinite(upper)||upper<128){$('benchmark-time').textContent='Adjust the prompt and output budgets to estimate timing.';$('benchmark-cost').textContent='Increase experiment context or reduce output tokens to fit a real prompt.';return;}
 if(!workloads){$('benchmark-time').textContent=search?`Context discovery only: quick load checks${full?', then one confirming long prompt':''}.`:'Select a workload or discover usable context.';$('benchmark-cost').textContent=search?`Per configuration: no speed samples, up to 12 quick load checks${full?' and up to 8 context probes':''}.`:'No cold speed-test workloads selected.';return;}
 const estimated=quick&&!full&&!hasSource&&Number($('output_tokens').value)===256&&Number.isInteger(repeats)&&repeats>=1&&repeats<=5;
 $('benchmark-time').textContent=(estimated?`Rough speed-test time: ${2*workloads*repeats}–${3*workloads*repeats} minutes per configuration.`:'Speed-test time: no estimate for these settings.')+(search?' Context-search time is additional.':'');
 $('benchmark-cost').textContent=`Per configuration: ${search?`context discovery first (up to 12 quick load checks${full?' and up to 8 context probes':''}), then `:''}${sizes.length*workloads*repeats} speed samples (${sizes.map(n=>n.toLocaleString()).join(' / ')} input tokens).`;
}
function resetExperiments(){
 for(const control of $('experiments').querySelectorAll('input')){
  if(control.type==='checkbox')control.checked=control.defaultChecked;
  else control.value=control.defaultValue;
 }
 resetExperimentCeiling();
 benchmarkCost();
 message('Experiment settings reset to defaults for the selected model.');
}
$('reset-experiments').onclick=resetExperiments;
for(const k of ['sweep_prompts','full_window','search_context','prompt_tokens','output_tokens','repeats'])$(k).addEventListener('input',benchmarkCost);
document.querySelectorAll('#workloads input').forEach(el=>el.addEventListener('input',benchmarkCost));
benchmarkCost();
function positionTip(tip){
 const content=tip.querySelector('.tip-content'), rect=tip.querySelector('button').getBoundingClientRect();
 const width=Math.min(280,window.innerWidth-24);
 content.style.left=Math.max(12,Math.min(rect.left,window.innerWidth-width-12))+'px';
 content.style.top=Math.max(12,Math.min(rect.bottom+6,window.innerHeight-content.offsetHeight-12))+'px';
}
function initTips(root=document){root.querySelectorAll('.tip').forEach(tip=>{
 if(tip.dataset.initialized)return;tip.dataset.initialized='true';
 const button=tip.querySelector('button');
 tip.addEventListener('mouseenter',()=>{tip.classList.remove('dismissed');positionTip(tip);});
 tip.addEventListener('mouseleave',()=>tip.classList.remove('dismissed'));
 button.addEventListener('focus',()=>{tip.classList.remove('dismissed');positionTip(tip);});
 button.addEventListener('click',e=>{e.preventDefault();const open=tip.classList.toggle('open');button.setAttribute('aria-expanded',String(open));tip.classList.toggle('dismissed',!open);if(!open)button.blur();else positionTip(tip);});
 button.addEventListener('keydown',e=>{if(e.key==='Escape'){tip.classList.remove('open');tip.classList.add('dismissed');button.setAttribute('aria-expanded','false');button.blur();}});
});}
launchTips();initTips();
for(const event of ['resize','scroll'])window.addEventListener(event,()=>document.querySelectorAll('.tip').forEach(tip=>{if(tip.querySelector('.tip-content').getClientRects().length)positionTip(tip);}),true);
document.addEventListener('click',e=>{document.querySelectorAll('.tip.open').forEach(tip=>{if(!tip.contains(e.target)){tip.classList.remove('open');tip.querySelector('button').setAttribute('aria-expanded','false');tip.querySelector('button').blur();}});});
let fileListing=null, selectedFile='', fileRequest=0;
function renderFiles(){
 const filter=$('file-filter').value.toLowerCase();
 const entries=(fileListing?.entries||[]).filter(e=>e.name.toLowerCase().includes(filter));
 $('file-list').replaceChildren();
 for(const entry of entries){
  const button=document.createElement('button');
  button.type='button';button.setAttribute('aria-pressed',String(selectedFile===entry.path));
  const name=document.createElement('span');name.textContent=(entry.directory?'📁 ':'')+entry.name;
  const detail=document.createElement('small');detail.textContent=entry.directory?'Folder':(entry.size/2**30).toFixed(2)+' GiB';
  button.append(name,detail);
  button.onclick=()=>{if(entry.directory){openFolder(entry.path);}else{selectedFile=entry.path;$('file-selected').textContent=entry.path;$('file-use').disabled=false;renderFiles();}};
  button.ondblclick=()=>{if(!entry.directory)useFile(entry.path);};
  $('file-list').append(button);
 }
 if(!entries.length)$('file-list').textContent='No matching folders or GGUF files.';
}
async function openFolder(directory){
 const request=++fileRequest;selectedFile='';fileListing=null;
 $('file-use').disabled=true;$('file-selected').textContent='No file selected.';
 $('file-list').textContent='Loading…';$('file-error').textContent='';
 try{
  const listing=await api('/api/files',{directory});
  if(request!==fileRequest||!$('file-dialog').open)return;
  fileListing=listing;$('file-directory').value=listing.directory;$('file-filter').value='';
  $('file-up').disabled=listing.parent===listing.directory;renderFiles();
 }catch(e){if(request===fileRequest){$('file-error').textContent=e.message;$('file-list').replaceChildren();}}
}
function closeFiles(){fileRequest++;$('file-dialog').close();}
function useFile(path){
 $('drafter').value=path;$('pair_confirmed').checked=false;
 closeFiles();edited('drafter');
}
$('browse-drafter').onclick=()=>{
 $('file-dialog').showModal();
 const current=$('drafter').value.trim(), slash=current.lastIndexOf('/');
 openFolder(slash>=0?(current.slice(0,slash)||'/'):null);
};
$('file-up').onclick=()=>{if(fileListing)openFolder(fileListing.parent);};
$('file-home').onclick=()=>openFolder(fileListing?.home||'~');
$('file-models').onclick=()=>openFolder(null);
$('file-go').onclick=()=>openFolder($('file-directory').value);
$('file-directory').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();openFolder(e.target.value);}};
$('file-filter').oninput=renderFiles;
$('file-use').onclick=()=>{if(selectedFile)useFile(selectedFile);};
for(const id of ['file-close','file-cancel'])$(id).onclick=closeFiles;
$('file-dialog').addEventListener('cancel',()=>{fileRequest++;});
$('export').onclick=()=>attempt(async()=>downloadText(JSON.stringify(await api('/api/results/export'),null,2),'application/json','lllm2-results.json'));
$('skip-content').onclick=e=>{e.preventDefault();const section=$(view+'-view');section.focus();section.scrollIntoView({block:'start'});};


// One editor, moved between views, with independent in-memory drafts and provenance.
function fill(s){
 savedFeedback='';
 if(s.model&&!Array.from($('model').options).some(o=>o.value===s.model))$('model').add(new Option(modelName(s.model),s.model));
 for(const k of settingKeys){
  if(k==='device')continue;
  if($(k).type==='checkbox')$(k).checked=s[k]??false;
  else $(k).value=s[k]??(k==='cuda_graph_opt'?'default':'');
 }
 $('device').replaceChildren(new Option(s.device||'Choose device',s.device||''));
 syncPlacement();
 slotNote();defaultState();launchState();
}
function rememberDraft(){drafts[view]={settings:settings(),defaults:loadedDefaults,engineOverride,savedExists,selectionNote};}
async function switchView(next){
 if(next==='find'){
  queuedView=null;
  $('launch-view').hidden=true;$('experiments-view').hidden=true;$('find-view').hidden=false;
  for(const name of ['launch','experiments','find'])$('nav-'+name).setAttribute('aria-current',name===next?'page':'false');
  $('skip-content').href='#find-view';
  await loadCatalogue();if(!findLoaded)await searchHF();return;
 }
 if(resolving||scanPending){queuedView=next;return;}
 $('find-view').hidden=true;
 $('launch-view').hidden=next!=='launch';$('experiments-view').hidden=next!=='experiments';
 for(const name of ['launch','experiments','find'])$('nav-'+name).setAttribute('aria-current',name===next?'page':'false');
 $('skip-content').href='#'+next+'-view';
 if(next===view)return;
 rememberDraft();selectionSequence++;validationSequence++;clearTimeout(editTimer);resolving=false;validationSnapshot='';
 view=next;
 $('skip-content').href='#'+next+'-view';
 if(!drafts[next])drafts[next]=structuredClone(drafts.launch);
 const d=drafts[next];loadedDefaults=d.defaults;engineOverride=d.engineOverride;savedExists=d.savedExists;selectionNote=d.selectionNote||'';
 $('launch-view').hidden=next!=='launch';$('experiments-view').hidden=next!=='experiments';
 $('nav-launch').setAttribute('aria-current',next==='launch'?'page':'false');$('nav-experiments').setAttribute('aria-current',next==='experiments'?'page':'false');
 $(next==='launch'?'editor-host-launch':'experiment-settings-host').append($('settings-editor'));
 $(next==='launch'?'feature-host-launch':'feature-host-experiments').append($('feature-availability'));
 $(next==='launch'?'summary-host-launch':'summary-host-experiments').append($('source-summary'));
 if(next==='launch')$('selection-note').before($('all-models'));else $('summary-host-experiments').before($('all-models'));
 $('customize-toggle').lastChild.textContent=next==='launch'?'Customize settings':'Customize experiment settings';
 $('save-default').hidden=next==='experiments';
 $('load-experiment').hidden=next==='experiments';
 $('load-experiment-note').hidden=next==='experiments';
 $('load-menu').open=false;
 fill(d.settings);renderRecommendations();await inspect();
 if(next==='experiments')await refreshResults();
}
for(const [id,next] of [['nav-launch','launch'],['nav-experiments','experiments'],['nav-find','find']])$(id).onclick=e=>{e.preventDefault();location.hash=next;attempt(()=>switchView(next));};
window.addEventListener('hashchange',()=>attempt(()=>switchView(location.hash==='#find'?'find':location.hash==='#experiments'?'experiments':'launch')));
function finishNavigation(){if(queuedView&&!resolving&&!scanPending){const next=queuedView;queuedView=null;attempt(()=>switchView(next));}}
function defaultState(){
 if(!$('summary-title'))return;
 const s=settings(),changed=loadedDefaults?settingKeys.filter(k=>settingValue(s,k)!==settingValue(loadedDefaults.settings,k)):[];
 const mode=changed.length?'custom':loadedDefaults?.mode||'custom';
 const names={recommended:'Recommended settings',saved:'My saved settings',custom:'Custom settings',running:'Current model settings',result:'Experiment settings · for next launch'};
 const source=loadedDefaults?.source||'Choose a model to resolve its starting settings.';
 const measured=source.startsWith('Measured built-in');
 $('source-summary').hidden=!s.model;
 $('selected-model').hidden=!s.model||view!=='launch';
 $('selected-model').textContent='Selected model: '+modelName(s.model);
 $('summary-title').textContent=resolving?'Preparing settings…':names[mode];
 $('summary-context').textContent=s.model?`${Math.floor(s.context/s.slots).toLocaleString()} tokens per conversation · ${s.slots} slot${s.slots===1?'':'s'} · ${s.backend}${mode==='recommended'?' · '+(measured?(source.includes('qualified')?'Qualified recommendation':'Tested on RTX 3090'):'Estimated starting settings'):''}`:'';
 $('draft-note').textContent=changed.length?'Changed: '+changed.map(k=>launchHelp[k][0]).join(', ')+'. Last loaded evidence does not validate these edits.':(s.speculation.includes('dflash')||s.speculation.includes(',')||cachePair(s)[0]!==cachePair(s)[1])?'Experimental settings selected. Review feature details before starting.':'';
 $('default-source').textContent=source;
 $('recommendation-notes').replaceChildren(...(loadedDefaults?.notes||[]).map(note=>{const p=document.createElement('p');p.textContent=note;return p;}));
 $('default-evidence').hidden=!loadedDefaults?.evidence;
 $('default-evidence').querySelector('summary').textContent=changed.length?'Last loaded evidence':'Recommendation evidence';
 $('default-evidence').querySelector('pre').textContent=loadedDefaults?.evidence?JSON.stringify(loadedDefaults.evidence,null,2):'';
 const loadingBlocked=resolving||scanPending||!connected;
 $('load-default').disabled=loadingBlocked||!savedExists;
 $('built-in-default').disabled=loadingBlocked||!s.model||!s.engine;
 $('load-experiment').disabled=loadingBlocked;
 $('saved-unavailable').textContent=!savedExists?'No saved settings for this model and backend.':'';
 $('recommended-unavailable').textContent=!s.model||!s.engine?'Choose a model and engine first.':'';
 $('settings-status').textContent=s.model?`Settings: ${names[loadedDefaults?.mode]||names.custom}${changed.length?' · Modified · Unsaved changes':savedFeedback?' · '+savedFeedback:mode==='result'?' · Not saved':''}`:'Settings will be prepared after you choose a model.';
 $('saved-note').textContent=savedExists&&mode==='recommended'?'Recommended settings are selected. Your saved settings are still available.':'';
 $('experiment-model').textContent=modelName(s.model);
 for(const id of ['drafter','pair_confirmed']){
  const el=id==='pair_confirmed'?$(id).closest('label').parentElement:$(id).closest('.wide');if(el)el.hidden=s.speculation!=='draft-dflash';
 }
}
function launchState(){
 const s=settings(), launch=currentLaunch(),running=statusState.engine||{},job=statusState.job||{};
 const same=sameSettings(launch,running.settings),launchJob=job.active&&job.kind==='launch',experiment=job.active&&job.kind!=='launch';
 $('cache-summary').textContent=`Actual attention cache: ${cacheLabel(s)}. ${cachePair(s)[0]!==cachePair(s)[1]?'Mixed precision is experimental; measured capacity and quality are not established.':''}`;
 $('batch-summary').textContent=`Prompt processing: logical batch ${s.batch_size??'engine default'} · microbatch ${s.ubatch_size??'engine default'}.`;
 $('reuse-summary').textContent=`Engine defaults: host cache ${featureState?.cache_ram_mib?.advertised_default??'unknown'} MiB; checkpoints ${featureState?.context_checkpoints?.advertised_default??'unknown'} per slot. Blank preserves these values for normal launches.`;
 $('execution-summary').textContent=`Target GPU sampling: ${s.backend_sampling?'requested':'engine default'} · concurrent CUDA streams: ${s.cuda_graph_opt}.`;
 const ready=connected&&!resolving&&validationSnapshot===snapshot()&&!pendingAction;
 let title='Start Model',status='Ready to start';
 if(!connected){title='Waiting for panel…';status='Panel connection unavailable. Last-known service state may be stale.';}
 else if(pendingAction||launchJob){title='Starting…';status=`Loading ${modelName(job.settings?.model||launch?.model)}${job.started_at?' · '+Math.max(0,Math.floor(Date.now()/1000-job.started_at))+'s elapsed':''}`;}
 else if(experiment){title='Experiment running';status='The experiment owns the engine. View its progress or cancel it in Experiments.';}
 else if(running.ready&&same){title='Running';status=`Ready · ${modelName(running.settings?.model)}`;}
 else if(resolving){title='Preparing model…';status='Checking model identity, recommended settings and engine compatibility…';}
 else if(!validationSnapshot&&!validationError&&launch?.model&&s.engine){title='Checking settings…';status='Checking the selected settings before starting.';}
 else if(running.running){title=launch?.model===running.settings?.model?'Restart with these settings':`Switch to ${modelName(launch?.model)}`;status='This replaces the running model and interrupts its current requests.';}
 else if(!launch?.model){status='Choose or download a model to get started.';}
 else if(validationError){status='Setup needs attention before starting.';}
 $('start').textContent=title;$('start').disabled=!ready||!!job.active||(running.ready&&same)||view!=='launch';
 $('start').hidden=connected&&!resolving&&!launch?.model&&!running.running&&!job.active;
 $('launch-status').textContent=status;
 $('stop').hidden=!(running.running||launchJob);$('stop').textContent=launchJob?'Cancel start':'Stop model';$('stop').disabled=!connected||pendingAction;
 $('copy-api').hidden=!running.ready;$('copy-api').disabled=!connected;
 $('connect-agent').hidden=!connected||!running.ready;
 $('resources-scope').textContent=connected?'Model workstation resources':'Model workstation resources · last known';
 $('connect-context').textContent=running.ready&&running.settings?`${modelName(running.settings.model)} · ${Math.floor(running.settings.context/running.settings.slots).toLocaleString()} tokens per conversation on the running server.`:'';
 $('running-summary').textContent=running.running&&running.settings?`Current model: ${modelName(running.settings.model)} · ${Math.floor(running.settings.context/running.settings.slots).toLocaleString()} tokens per conversation · ${running.ready?'ready':'loading'}`:'';
 const rawError=actionError||running.error||(job.status==='failed'?job.error:'')||(view==='launch'&&s.model?validationError:'')||'';
 const friendlyError=explainError(rawError);
 $('launch-error').textContent=friendlyError;
 $('error-details').hidden=!rawError;$('error-raw').textContent=rawError;
 $('experiment-error').textContent=view==='experiments'?(!connected?'Panel connection unavailable. Reconnect before running an experiment.':validationError||actionError||running.error||''):'';
 $('setup-actions').hidden=resolving||(!validationError&&connected&&!!launch?.model);
 $('browse-models').hidden=!!launch?.model;
 $('engine-setup').hidden=resolving||(!!discovered.engines?.some(e=>e.devices?.length)&&!!s.engine&&!/engine|llama-server|binary|backend|device/i.test(validationError));
 $('selection-note').textContent=selectionNote;
 $('launch-state').textContent=running.running&&!same?'Settings shown are for the next start. The current model is unchanged.':'';
 $('global-operation').hidden=!job.active;
 renderMarkup('global-operation',job.active?`${esc(launchJob?'Model starting':'Experiment running')} · ${esc(job.phase||job.status)} <a id="operation-progress" href="#${launchJob?'launch':'experiments'}">View progress</a>`:'');
 document.querySelectorAll('.run').forEach(b=>{b.disabled=!ready||!!job.active||view!=='experiments'||(b.dataset.mode==='combinations'&&!combinations.length);b.textContent=(running.running?'Stop model and run · ':'')+b.dataset.label;});
 $('cancel').disabled=!connected||!job.active||pendingAction;
 $('save-default').disabled=!ready||view!=='launch';
 modelControlsState();
 const loadingBlocked=resolving||scanPending||!connected;
 $('load-default').disabled=loadingBlocked||!savedExists;
 $('built-in-default').disabled=loadingBlocked||!s.model||!s.engine;
 $('load-experiment').disabled=loadingBlocked;
}
function explainError(error){
 if(/out of memory|cuda.*alloc|failed to allocate|insufficient.*memory/i.test(error))return 'Not enough memory for this configuration. Close unused applications, choose Auto GPU placement, or reduce context under Customize, then try again.';
 if(/exceed.*context|context.*exceed|context.*overflow|prompt.*too long/i.test(error))return 'The request exceeds the context window. Increase context and restart the model, or shorten the conversation. Relaunch your agent after changing context.';
 if(/address already in use|port.*occupied|port.*in use/i.test(error))return 'The engine port is occupied. Check Engine logs and stop the conflicting service yourself, or configure a different engine port.';
 if(/no such file|not found|does not exist|disappear/i.test(error))return 'A selected model or engine file is unavailable. Check its path under Customize or rescan model locations.';
 return error.length>240?'The operation could not complete. Open the exact error details and engine logs, check the selected settings, then try again.':error;
}
function renderHardware(hardware){
 const gpus=hardware.gpus||[],fmt=n=>Number.isFinite(n)?n.toFixed(1):'—';
 $('gpu').textContent=gpus.length?gpus.map(g=>g.name).join(' · '):'No NVIDIA GPU detected';
 $('vram').textContent=gpus.length?gpus.map((g,i)=>`${gpus.length>1?'GPU '+(g.index??i)+' ':''}VRAM ${fmt(Number.isFinite(g.used_mib)?g.used_mib/1024:NaN)} / ${fmt(Number.isFinite(g.total_mib)?g.total_mib/1024:NaN)} GiB`).join(' · '):'VRAM unavailable';
 const ram=hardware.ram||{};
 $('ram').textContent=`RAM ${fmt(ram.used_gib)} / ${fmt(ram.total_gib??hardware.ram_gib)} GiB`;
 $('ram-available').textContent=Number.isFinite(ram.available_gib)?`${fmt(ram.available_gib)} GiB RAM available to applications.`:'Available RAM could not be read.';
}
const pendingDownloads=new Set();
function modelControlsState(){
 for(const b of document.querySelectorAll('[data-select-model],[data-download],[data-verify]')){
  b.disabled=!connected||pendingAction||pendingDownloads.has(b.dataset.download)||(b.dataset.selectModel?(resolving||scanPending||b.dataset.selectModel===currentLaunch()?.model):false);
 }
}
function renderMarkup(id,markup){
 const root=$(id);if(root._markup===markup)return;root._markup=markup;
 const focused=root.contains(document.activeElement)?document.activeElement.id:null;
 const open=[...root.querySelectorAll('details[open]')].map(d=>d.id);
 root.innerHTML=markup;
 for(const id of open){const el=$(id);if(el)el.open=true;}
 if(focused)($(focused)||$('find-more'))?.focus({preventScroll:true});
}
function setModelFilter(filter){if(filter==='catalog'){location.hash='find';attempt(()=>switchView('find'));return;}renderRecommendations();}
function renderRecommendations(){
 const catalog=discovered.catalog||[],models=discovered.models||[],selected=currentLaunch()?.model;
 const rank=m=>catalog.find(c=>c.id===m.catalog_id)?.recommendation?.rank||999;
 const rows=[...models].sort((a,b)=>rank(a)-rank(b)||a.path.localeCompare(b.path));
 $('installed-count').textContent=`${models.length} installed`;
 $('recommended-list').hidden=false;$('catalog').hidden=false;
 const badge=c=>c?.recommendation?` <span class="pill">${esc(c.recommendation.label)}</span>`:'';
 const details=(id,text)=>`<details id="${esc(id)}"><summary id="${esc(id)}-summary">Variant & location</summary>${esc(text)}</details>`;
 renderMarkup('recommended-list',rows.map(m=>{
  const c=catalog.find(c=>c.id===m.catalog_id),recommended=m.identity_verified?c:null;
  return `<div class="model-choice ${selected===m.path?'selected':''}"${selected===m.path?' aria-current="true"':''}><div><b>${esc(modelName(m.path))}</b>${badge(recommended)}<small>${esc(c?.recommendation?.variant||c?.file||m.path.split('/').pop())} · Installed</small>${recommended?.recommendation?`<small>${esc(recommended.recommendation.description)}</small>`:''}${details('installed-detail-'+encodeURIComponent(m.path),m.path+(m.identity_verified?' · Verified checkpoint identity':' · No measured identity verified'))}</div>${selected===m.path?`<span class="selected-label">✓ Selected<span class="visually-hidden">: ${esc(modelName(m.path))}</span></span>`:`<button id="installed-use-${esc(encodeURIComponent(m.path))}" aria-label="Use model: ${esc(modelName(m.path))} · ${esc(m.path)}" data-select-model="${esc(m.path)}">Use this model</button>`}</div>`;
 }).join('')||'<p class="muted">No installed models found. Browse the catalogue to download one, or check your model locations below.</p><button id="empty-browse">Browse models</button>');
 renderCatalogue();
 $('model-browser-note').textContent='';
 modelControlsState();
}
function renderDownloads(){
 if(!discovered.catalog)return;
 const ds=statusState.downloads||[];
 $('download-section').hidden=!ds.length;
 renderMarkup('active-downloads',ds.map(d=>`<div class="download-row" id="download-${esc(d.id)}"><div class="row"><div><h3>${esc(d.name)}</h3><small data-download-status></small></div><button id="download-action-${esc(d.id)}"></button></div><progress aria-label="${esc(d.name)} download progress"></progress><details id="download-detail-${esc(d.id)}"><summary id="download-summary-${esc(d.id)}">Download details</summary><small data-download-detail></small></details></div>`).join(''));
 for(const d of ds){
  const row=$('download-'+d.id),active=['queued','downloading'].includes(d.state),button=$('download-action-'+d.id);
  row.querySelector('[data-download-status]').textContent=`${d.state==='complete'?'Downloaded':d.state} · ${d.done_gb??0} / ${d.total_gb||'unknown'} GB${active?` · ${d.rate_mib_s??0} MiB/s`:''}`;
  row.querySelector('[data-download-detail]').textContent=[d.target,d.file,d.detail].filter(Boolean).join(' · ');
  const progress=row.querySelector('progress');progress.max=100;if(d.total_gb)progress.value=Math.min(100,Math.max(0,d.percent||0));else progress.removeAttribute('value');progress.hidden=!active;
  delete button.dataset.selectModel;delete button.dataset.download;delete button.dataset.active;
  if(d.state==='complete'){
   button.hidden=true;delete button.dataset.verify;
  }else{button.hidden=false;delete button.dataset.verify;button.textContent=active?'Cancel download':'Retry download';button.dataset.download=d.id;button.dataset.active=String(active);}
  button.setAttribute('aria-label',button.textContent+' · '+d.name);
  button.disabled=!connected||resolving||scanPending||pendingAction;
 }
 const notice=ds.some(d=>d.state==='complete')?'Download complete. Open Launch or Experiments to use the model.':ds.some(d=>d.state==='error')?'A download failed. Open its details, then retry.':ds.some(d=>['queued','downloading'].includes(d.state))?'Downloading to the model workstation. You can continue using the panel.':'';
 if($('download-notice').textContent!==notice)$('download-notice').textContent=notice;
 renderRecommendations();
}
async function scan(){
 if(scanPending)return false;
 let applied=false;scanPending=true;const n=++selectionSequence;resolving=true;validationSnapshot='';defaultState();launchState();
 const before=settings();
 try{
  const data=await api('/api/discover',{});
  if(n!==selectionSequence)return;
  discovered=data;applied=true;
  $('model').replaceChildren(new Option('Choose an installed model',''),...data.models.map(m=>new Option(`${modelName(m.path)} · ${m.path}`,m.path)));
  $('engine-list').replaceChildren(...data.engines.map(e=>{const o=new Option(e.path,e.path);o.label=e.devices?.join(', ')||'No GPU device';return o;}));
  renderDownloads();
  const initial=!discoveredOnce;discoveredOnce=true;
  const running=statusState.engine?.running?statusState.engine.settings:statusState.job?.kind==='launch'&&statusState.job?.active?statusState.job.settings:null;
  if(initial&&running){loadedDefaults={settings:running,mode:'running',source:'Current model settings',notes:['Showing the current operation. No saved preferences changed.']};fill(running);await inspect();}
  else if(before.model){fill(before);await inspect();}
  else if(initial){await selectModel('');}
  else{fill(before);await inspect();}
 }catch(e){if(n===selectionSequence)validationError='Discovery failed: '+e.message;}
 finally{scanPending=false;if(n===selectionSequence)resolving=false;defaultState();renderDownloads();launchState();finishNavigation();}
 return applied;
}
async function selectModel(path){
 const n=++selectionSequence;validationSequence++;resolving=true;validationSnapshot='';validationError='';selectionNote='';actionError='';loadedDefaults=null;savedExists=false;
 if(path){if(!Array.from($('model').options).some(o=>o.value===path))$('model').add(new Option(modelName(path),path));$('model').value=path;}
 defaultState();launchState();
 try{
  const data=await api('/api/launch/select',{model:path,...(engineOverride?{engine:$('engine').value,backend:$('backend').value,device:$('device').value}:{})});
  if(n!==selectionSequence)return;
  selectionNote=data.reason||'';
  if(data.settings){loadedDefaults={...data,mode:'recommended'};fill(data.settings);await inspect();}
  else{validationError=data.reason;}
  if(n===selectionSequence&&(view==='experiments'||!drafts.experiments))resetExperimentCeiling();
 }catch(e){if(n===selectionSequence)validationError=e.message;}
 finally{if(n===selectionSequence){resolving=false;defaultState();renderRecommendations();launchState();finishNavigation();}}
}
async function loadDefaults(source='built-in'){
 const n=++selectionSequence,selected=settings();validationSequence++;resolving=true;validationSnapshot='';validationError='';defaultState();launchState();
 try{
  const d=await api('/api/default/resolve',{settings:selected,source});
  if(n!==selectionSequence)return;
  loadedDefaults={...d,mode:source==='saved'?'saved':'recommended'};fill(d.settings);await inspect();
 }catch(e){if(n===selectionSequence)validationError='Settings could not be loaded: '+e.message;}
 finally{if(n===selectionSequence){resolving=false;defaultState();launchState();finishNavigation();}}
}
async function inspect(){
 const n=++validationSequence,selected=settings(),key=JSON.stringify(selected);validationSnapshot='';validationError='';featureState=null;
 $('engine-alternative').hidden=true;
 $('model-facts').textContent=(discovered.models?.find(m=>m.path===selected.model)?.metadata.context?'Checkpoint metadata context limit: '+discovered.models.find(m=>m.path===selected.model).metadata.context.toLocaleString()+' tokens. ':'')+'Usable allocation is shown above.';
 launchState();
 if(!selected.model||!selected.engine){validationError=!selected.model?'Choose or download a model.':'Choose an installed GPU-enabled llama-server under Customize settings.';launchState();return;}
 $('features').textContent='Checking support…';
 try{
  const [c,v]=await Promise.all([api('/api/capabilities',{settings:selected}),api('/api/launch/check',{settings:selected})]);
  if(n!==validationSequence||key!==snapshot())return;
  featureState=c.features;savedExists=v.saved_exists;
  const devices=c.engine.devices.filter(d=>d.startsWith(selected.backend));
  $('device').replaceChildren(...[...new Set([selected.device,...devices])].map(d=>new Option(d||'Choose device',d)));
  $('device').value=selected.device;
  $('device-hint').textContent=devices.includes(selected.device)?'':`Choose a detected ${selected.backend} device or a matching engine build.`;
  $('device-probe').hidden=devices.includes(selected.device);$('device-output').textContent=c.engine.error||c.engine.device_output||'';
  const alternative=discovered.engines?.find(e=>e.path!==selected.engine&&e.devices?.some(d=>d.startsWith(selected.backend)));
  if(!devices.length&&alternative){$('engine-alternative').hidden=false;$('use-backend-engine').textContent=`Use matching ${selected.backend} build`;$('backend-engine-path').textContent=alternative.path;$('use-backend-engine').onclick=()=>attempt(async()=>{
    if(key!==snapshot())return;
    $('engine').value=alternative.path;$('device').replaceChildren(new Option(alternative.devices.find(d=>d.startsWith(selected.backend))));engineOverride=true;await loadDefaults();
  });}
  // Methods needing setup must remain selectable so their prerequisite controls can be reached.
  for(const option of $('speculation').options)option.disabled=false;
  for(const k of ['flash','cache','effort'])$(k).disabled=false;
  validationError=v.error||'';if(v.valid)validationSnapshot=key;
  renderFeatures();defaultState();launchState();
 }catch(e){if(n===validationSequence&&key===snapshot()){validationError=e.message;$('features').textContent=e.message;launchState();}}
}
function edited(key){
 savedFeedback='';
 if(key==='gpu_layers')syncPlacement();
 selectionSequence++;validationSequence++;resolving=false;validationSnapshot='';validationError='';actionError='';clearTimeout(editTimer);
 if(key==='cache')for(const k of optionalCache)$(k).value='';
 if(key==='drafter')$('pair_confirmed').checked=false;
 if(['engine','backend','device'].includes(key))engineOverride=true;
 defaultState();slotNote();renderFeatures();launchState();
 editTimer=setTimeout(()=>attempt(inspect),250);finishNavigation();
}
async function poll(){
 if(pollPending)return;
 pollPending=true;
 try{
  const s=await api('/api/status');statusState=s;token=s.token;connected=true;
  $('app-version').textContent=s.version?`Version ${s.version}`:'';
  if(startAttempt&&s.job.request_id===startAttempt.request_id&&['failed','cancelled','serving'].includes(s.job.status))startAttempt=null;
  renderHardware(s.hardware);
  $('endpoint').textContent=s.engine.ready?`API on model workstation: ${s.endpoint}`:'';
  $('paths').textContent=`Models: ${s.paths.models}. Engine roots: ${s.paths.engines.join(', ')}. Configure LLLM2_MODELS_DIR and LLLM2_ENGINE_ROOTS before starting the panel.`;
  $('job').textContent=`${s.job.status}${s.job.current?' · '+s.job.current:''}`;
  $('job-detail').textContent=[s.job.phase,s.job.error,...(s.job.skipped||[]).map(x=>`${x.option}: ${x.reason}`)].filter(Boolean).join(' · ');
  $('bar').style.width=s.job.total?`${100*(s.job.completed||0)/s.job.total}%`:'0%';
  $('logs').textContent=[(s.engine.argv||[]).join(' '),'',...(s.engine.logs||[])].join('\n');
  renderDownloads();launchState();resultDeleteControls();
  const complete=(s.downloads||[]).filter(d=>d.state==='complete').map(d=>d.id).sort().join('|');
  if(complete!==lastDownloads){if(!discoveredOnce)lastDownloads=complete;else if(!scanPending&&await scan())lastDownloads=complete;}
  if(view==='experiments'){try{await refreshResults();}catch(e){message('Experiment history could not be refreshed: '+e.message,true);}}
 }catch(e){connected=false;launchState();}
 finally{pollPending=false;}
}
$('refresh').onclick=()=>attempt(scan);
$('browse-models').onclick=()=>{setModelFilter('catalog');$('nav-find').focus();};
$('retry-setup').onclick=()=>attempt(async()=>{await poll();await scan();});
function customize(open){$('customize').hidden=!open;$('customize-toggle').setAttribute('aria-expanded',String(open));$('customize-toggle').querySelector('.marker').textContent=open?'▾':'▸';}
$('customize-toggle').onclick=()=>customize($('customize').hidden);
function syncPlacement(){const auto=$('gpu_layers').value==='';$('gpu-placement').value=auto?'auto':'manual';$('gpu_layers').disabled=auto;}
let manualLayers=999;
$('gpu-placement').onchange=()=>{
 if($('gpu-placement').value==='auto'){manualLayers=Number($('gpu_layers').value);$('gpu_layers').value='';}
 else $('gpu_layers').value=manualLayers;
 syncPlacement();edited('gpu_layers');
};
for(const id of ['choose-engine','manage-engine'])$(id).onclick=()=>{customize(true);$('engine').focus();};
$('model').onchange=()=>attempt(()=>selectModel($('model').value));
for(const k of settingKeys){if(k!=='model')$(k).addEventListener('input',()=>edited(k));}
for(const k of ['engine','backend'])$(k).onchange=()=>{clearTimeout(editTimer);attempt(loadDefaults);};
$('load-default').onclick=()=>{$('load-menu').open=false;attempt(()=>loadDefaults('saved'));};
$('built-in-default').onclick=()=>{$('load-menu').open=false;attempt(()=>loadDefaults('built-in'));};
document.addEventListener('click',e=>{if(!$('load-menu').contains(e.target))$('load-menu').open=false;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('load-menu').open){$('load-menu').open=false;$('load-menu').querySelector('summary').focus();}});
$('save-default').onclick=()=>attempt(async()=>{
 const selected=settings(),key=snapshot(),savedView=view,provenance=loadedDefaults?.mode==='result'&&sameSettings(selected,loadedDefaults.settings)?loadedDefaults:null;
 const saved=await api('/api/default/save',provenance?{result_id:provenance.result_id,use_context:provenance.use_context,reserve_headroom:provenance.reserve_headroom}:{settings:selected});
 if(key===snapshot()&&savedView===view){loadedDefaults={settings:saved,mode:'saved',source:provenance?'My saved settings · experiment evidence':'My saved settings · manual preferences',notes:provenance?.notes||['Manual preferences are not a measured benefit.'],evidence:provenance?.evidence};savedExists=true;savedFeedback='Saved';defaultState();}
 message('Settings saved for this model and backend.');
});
$('start').onclick=()=>attempt(async()=>{
 if($('start').disabled||view!=='launch')return;
 const selected=structuredClone(settings()),running=statusState.engine||{};
 if(!startAttempt||!sameSettings(startAttempt.settings,selected))startAttempt={request_id:crypto.randomUUID?crypto.randomUUID():`${Date.now()}-${Math.random()}`,settings:selected};
 pendingAction=true;actionError='';launchState();
 try{await api('/api/start',{...startAttempt,replace_running:!!running.running,expected_pid:running.pid});startAttempt=null;}
 catch(e){actionError='Start could not be confirmed: '+e.message+'. Check service status before retrying.';}
 finally{await poll();pendingAction=false;launchState();}
});
for(const k of ['stop','cancel'])$(k).onclick=()=>attempt(async()=>{
 if(pendingAction)return;pendingAction=true;launchState();
 try{await api('/api/'+k,{});actionError='';}finally{await poll();pendingAction=false;launchState();}
});
async function copyText(text,label,workstation=true){
 try{await navigator.clipboard.writeText(text);message(label+' copied.'+(workstation?' Use it on the model workstation.':''));}
 catch{$('copy-title').textContent=workstation?'Copy on the model workstation':'Copy '+label.toLowerCase();$('copy-text').rows=workstation?1:10;$('copy-text').value=text;$('copy-dialog').showModal();$('copy-text').focus();$('copy-text').select();}
}
$('copy-api').onclick=()=>attempt(()=>copyText(statusState.endpoint,'API address'));
$('copy-agent').onclick=()=>attempt(()=>copyText($('agent-command').textContent,'Agent command'));
$('copy-close').onclick=()=>$('copy-dialog').close();
document.querySelectorAll('[data-copy]').forEach(b=>b.onclick=()=>attempt(()=>copyText(b.dataset.copy,'Command')));
document.querySelectorAll('[data-agent]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('[data-agent]').forEach(other=>other.setAttribute('aria-pressed',String(other===b)));
 $('agent-command').textContent='lllm2 '+b.dataset.agent;
});
async function downloadClick(e){
 if(e.target.closest('[data-verify]')){await scan();return;}
 if(e.target.closest('#empty-browse')){setModelFilter('catalog');$('nav-find').focus();return;}
 const variant=e.target.closest('[data-select-model]');
 if(variant){if(variant.disabled)return;await selectModel(variant.dataset.selectModel);return;}
 const b=e.target.closest('[data-download]');if(!b||b.disabled)return;
 const id=b.dataset.download;pendingDownloads.add(id);modelControlsState();
 try{await api(b.dataset.active==='true'?'/api/download/cancel':'/api/download',{id});await poll();}
 finally{pendingDownloads.delete(id);renderDownloads();}
}
for(const id of ['recommended-list','active-downloads'])$(id).onclick=e=>attempt(()=>downloadClick(e));
$('use-launch-settings').onclick=()=>attempt(async()=>{
 if(view!=='experiments')return;
 selectionSequence++;clearTimeout(editTimer);resolving=false;const d=structuredClone(drafts.launch);loadedDefaults=d.defaults;engineOverride=d.engineOverride;fill(d.settings);resetExperimentCeiling();await inspect();
});
document.querySelectorAll('.run').forEach(b=>{
 b.dataset.label=b.textContent;
 b.onclick=()=>attempt(async()=>{
  if(b.disabled||view!=='experiments')return;
  const data={settings:structuredClone(settings()),mode:b.dataset.mode,combinations:structuredClone(combinations),replace_running:!!statusState.engine?.running,expected_pid:statusState.engine?.pid,workloads:[...document.querySelectorAll('#workloads input:checked')].map(x=>x.value),search_context:$('search_context').checked,sweep_prompts:$('sweep_prompts').checked,full_window:$('full_window').checked};
  for(const k of ['prompt_tokens','output_tokens','timeout','context_timeout','repeats','max_context'])data[k]=Number($(k).value);
  if(data.mode==='warm-conversation'){data.search_context=false;data.sweep_prompts=false;data.full_window=false;data.workloads=['long-code'];}
  pendingAction=true;launchState();
  try{await api('/api/benchmark',data);message('Experiment queued. Partial results are saved.');}finally{await poll();pendingAction=false;launchState();}
 });
});
function renderCombinations(){
 $('combo-count').textContent=`${combinations.length} selected`;
 $('combos').textContent=combinations.length?combinations.map((s,i)=>`${i+1}. ${modelName(s.model)} · ${s.speculation} / ${cacheLabel(s)} / flash ${s.flash} / effort ${s.effort} / draft ${s.draft_length}`).join('\n'):'No combinations selected. Add the current settings to begin.';
 launchState();
}
$('add-combo').onclick=()=>{combinations.push(structuredClone(settings()));renderCombinations();};
$('clear-combos').onclick=()=>{combinations=[];renderCombinations();};
let contextChoice='tested';
function headroomControl(r,key){
 const name='ctx-'+key,fmt=resultFormat;
 const radio=(choice,label)=>`<label><input type="radio" name="${esc(name)}" data-context-choice="${choice}" ${contextChoice===choice?'checked':''}>${label}</label>`;
 return `<span class="muted">Context:</span>${radio('tested',`${r.context_confirmed===false?'Loaded':'Tested'} (${fmt(r.largest_observed_context)})`)}${radio('headroom',`90% (${fmt(r.recommended_context)})`)}${radio('original','Original')}`;
}
document.addEventListener('change',e=>{
 if(!e.target.matches('[data-context-choice]')||!e.target.checked)return;
 contextChoice=e.target.dataset.contextChoice;
 document.querySelectorAll('[data-context-choice]').forEach(input=>input.checked=input.dataset.contextChoice===contextChoice);
});
async function previewResult(id,useContext=false){
 const n=++selectionSequence;validationSequence++;resolving=true;validationSnapshot='';launchState();
 try{
  const d=await api('/api/result/preview',{result_id:id,use_context:useContext,...(useContext?{reserve_headroom:contextChoice==='headroom'}:{})});
  if(n!==selectionSequence)return;
  const expected=n+(view==='launch'?0:1);resolving=false;await switchView('launch');
  if(selectionSequence!==expected)return;
  location.hash='launch';loadedDefaults={...d,mode:'result'};fill(d.settings);await inspect();
  message('Experiment settings copied to Launch. Review before starting or saving; your saved settings are unchanged.');
 }finally{if(n===selectionSequence){resolving=false;defaultState();launchState();finishNavigation();}}
}
$('results').onclick=e=>{
 const remove=e.target.closest('[data-delete-result]');if(remove){if(!remove.disabled)openResultDeletion([results.find(r=>r.id===remove.dataset.deleteResult)].filter(Boolean),false);return;}
 const expand=e.target.closest('[data-expand]');
 if(expand){
  const key=expand.dataset.expand,open=!expandedResults.has(key);if(open)expandedResults.add(key);else expandedResults.delete(key);
  expand.setAttribute('aria-expanded',String(open));expand.setAttribute('aria-label',expand.getAttribute('aria-label').replace(/^(Show|Hide)/,open?'Hide':'Show'));expand.textContent=open?'▾':'▸';$('detail-'+key).hidden=!open;return;
 }
 const copy=e.target.closest('[data-copy-row]');if(copy){const row=resultRows().find(r=>r.key===copy.dataset.copyRow);if(row)attempt(()=>copyText(tableText([row],'\t'),'Result row',false));return;}
 const b=e.target.closest('[data-promote]');if(b)attempt(()=>previewResult(b.dataset.promote,b.dataset.hasContext==='true'&&contextChoice!=='original'));
};
function resultBlock(r){return r.measurement_mode==='warm-conversation'?'Warm-only evidence':r.quality_status==='failed'?'Source adherence failed':r.status!=='complete'?'Run not complete':!r.samples?.length&&!r.largest_observed_context?'No completed samples or context measurement':'';}
$('load-experiment').onclick=()=>attempt(async()=>{
 $('load-menu').open=false;$('experiment-picker').showModal();$('experiment-picker-error').textContent='';$('experiment-options').textContent='Loading experiments…';
 const n=++pickerSequence;
 try{
  const rows=await api('/api/results');if(n!==pickerSequence||!$('experiment-picker').open)return;
  $('experiment-options').innerHTML=rows.map(r=>{const reason=resultBlock(r);return `<div class="experiment-option"><b>${esc(r.label||'Experiment')}</b><small>${esc(r.started)} · ${esc(r.settings.backend)} · ${Number(r.settings.context).toLocaleString()} total tokens</small><small>${esc([...new Set(r.samples.map(s=>s.workload).filter(Boolean))].join(', ')||r.measurement_mode||'Cold benchmark')}</small><small>${esc(r.settings.model)}</small><small>${reason?esc(reason):'Completed experiment · execution evidence'}</small><span class="context-load-actions"><button data-result="${esc(r.id)}" data-has-context="${!!r.largest_observed_context}" ${reason?'disabled':''}>Try in Launch</button>${!reason&&r.largest_observed_context?headroomControl(r,'picker-'+encodeURIComponent(r.id)):''}</span></div>`;}).join('')||'<p>No experiments yet. Run an experiment from the Experiments view, then return here.</p>';
 }catch(e){if(n===pickerSequence){$('experiment-options').textContent='';$('experiment-picker-error').textContent='Could not load experiments. Close and try again. '+e.message;}}
});
$('experiment-picker-close').onclick=()=>{pickerSequence++;$('experiment-picker').close();};
$('experiment-picker').addEventListener('cancel',()=>pickerSequence++);
$('experiment-options').onclick=e=>{const b=e.target.closest('[data-result]');if(!b||b.disabled)return;$('experiment-picker').close();attempt(()=>previewResult(b.dataset.result,b.dataset.hasContext==='true'&&contextChoice!=='original'));};
// Find models keeps discovery and catalogue actions separate from launch drafts.
let findLoaded=false,findBusy=false,findEntries=[],removingId=null;
let findSort={key:null,direction:0};
const findColumns=[['display_name','Model'],['repo','Publisher / repository'],['quant','Quantisation'],['size_gb','Size GB','number'],['fit','Suitability'],['task','Task'],['downloads','Downloads','number'],['likes','Likes','number'],['updated','Updated'],['license','Licence']];
$('find-table').querySelector('thead').innerHTML='<tr>'+findColumns.map(([key,label])=>`<th scope="col" aria-sort="none"><button type="button" data-find-sort="${key}"><span>${label}</span><span aria-hidden="true">↕</span></button></th>`).join('')+'<th scope="col">Catalogue</th></tr><tr class="find-filter-row">'+findColumns.map(([key,label,type])=>`<td>${type==='number'?`<div class="find-number-filter"><input type="number" min="0" step="any" data-find-min="${key}" aria-label="Minimum ${label}" placeholder="Min"><input type="number" min="0" step="any" data-find-max="${key}" aria-label="Maximum ${label}" placeholder="Max"></div>`:`<input type="search" data-find-filter="${key}" aria-label="Filter ${label}" aria-describedby="find-filter-help" title="Space-separated terms must all match. Prefix ! to exclude; use quotes for phrases." placeholder="Filter ${label.toLowerCase()}">`}</td>`).join('')+'<td></td></tr>';

function findTextTerms(query){
 // An unfinished quoted phrase remains usable while typing; a lone ! is ignored.
 return [...query.matchAll(/(!?)(?:"([^"]*)(?:"|$)|([^\s"]+))/g)]
  .map(match=>({exclude:match[1]==='!',text:(match[2]??match[3]).toLowerCase()}))
  .filter(term=>term.text&&term.text!=='!');
}
function filteredFindEntries(){
 const textFilters=[...$('find-table').querySelectorAll('[data-find-filter]')].map(input=>({key:input.dataset.findFilter,terms:findTextTerms(input.value)}));
 return findEntries.filter(e=>{
  if($('find-suitable').checked&&e.fit_rank>1)return false;
  if($('find-instruct').checked&&!e.instruct)return false;
  if($('find-quants').checked&&!/^(?:I?Q)[4-8]/.test(e.quant))return false;
  for(const {key,terms} of textFilters){
   const value=String(e[key]??'').toLowerCase();
   if(!terms.every(term=>term.exclude?!value.includes(term.text):value.includes(term.text)))return false;
  }
  for(const input of $('find-table').querySelectorAll('[data-find-min],[data-find-max]')){
   if(input.value==='')continue;
   const key=input.dataset.findMin||input.dataset.findMax,value=e[key];
   if(value==null||(input.dataset.findMin?value<Number(input.value):value>Number(input.value)))return false;
  }
  return true;
 }).sort((a,b)=>{
  if(!findSort.key)return 0;
  const av=a[findSort.key],bv=b[findSort.key];
  if(av==null||bv==null)return av==null?(bv==null?0:1):-1;
  const comparison=typeof av==='number'?av-bv:String(av).localeCompare(String(bv));
  return comparison*findSort.direction||(findSort.key==='fit_rank'?(b.downloads-a.downloads||b.updated.localeCompare(a.updated)):0)||a.repo.localeCompare(b.repo)||a.file.localeCompare(b.file);
 });
}
function renderFind(){
 const rows=filteredFindEntries();
 for(const button of $('find-table').querySelectorAll('[data-find-sort]')){
  const key=button.dataset.findSort,active=key===findSort.key;
  button.parentElement.setAttribute('aria-sort',active?(findSort.direction===1?'ascending':'descending'):'none');
  const label=findColumns.find(c=>c[0]===key)[1],next=active?(findSort.direction===1?'Sort descending':'Clear sort'):'Sort ascending';
  button.lastElementChild.textContent=active?(findSort.direction===1?'↑':'↓'):'↕';
  button.title=next+' · '+label;button.setAttribute('aria-label',label+': '+next.toLowerCase());
 }
 $('find-table').querySelector('tbody').innerHTML=rows.map(e=>{
  const saved=(discovered.catalog||[]).some(c=>c.repo===e.repo&&c.file===e.file);
  return `<tr${saved?' class="find-in-catalogue"':''}>`+findColumns.map(([key])=>`<td>${key==='display_name'?`<a href="https://huggingface.co/${esc(e.repo)}" target="_blank" rel="noopener noreferrer">${esc(e.display_name)}</a>${saved?'<span class="find-catalogue-badge">In catalogue</span>':''}<small>${esc(e.file)}</small>`:key==='fit'?`${esc(e.fit)}<small>${esc(e.reason)}</small>`:esc(key==='updated'?e.updated.slice(0,10):e[key]??'Unknown')}</td>`).join('')+`<td><button data-find-add="${esc(e.id)}" ${saved||e.issue?'disabled':''}>${saved?'In catalogue':'Add to catalogue'}</button>${e.issue?`<small>${esc(e.issue)}</small>`:''}</td></tr>`;
 }).join('')||'<tr><td colspan="11">No matching variants. Try a different search or relax the filters.</td></tr>';
 $('find-count').textContent=`${rows.length} of ${findEntries.length} variants shown.`;
}
async function searchHF(refresh=false){
 if(findBusy)return;findBusy=true;$('find-search').disabled=$('find-refresh').disabled=true;
 $('find-status').textContent='Reading Hugging Face metadata…';
 try{
  const result=await api('/api/models/find',{query:$('find-query').value,refresh});
  findEntries=result.entries.sort((a,b)=>a.fit_rank-b.fit_rank||b.downloads-a.downloads||b.updated.localeCompare(a.updated)||a.repo.localeCompare(b.repo));findLoaded=true;renderFind();
  $('find-status').textContent=`${result.repositories} repositories inspected · metadata fetched ${new Date(result.fetched_at*1000).toLocaleString()}. ${result.warning||''}`;
 }catch(e){$('find-status').textContent=e.message;}
 finally{findBusy=false;$('find-search').disabled=$('find-refresh').disabled=false;}
}
async function loadCatalogue(){
 const result=await api('/api/catalogue',{});discovered.catalog=result.entries;renderRecommendations();renderFind();
}
function renderCatalogue(){
 const entries=discovered.catalog||[];
 renderMarkup('catalog',entries.map(e=>{
  const job=(statusState.downloads||[]).find(d=>d.id===e.id),active=job&&['queued','downloading'].includes(job.state);
  const installed=e.installed===true||(e.installed===undefined&&job?.state==='complete');
  return `<div class="model-choice"><div><b>${esc(e.display_name||e.name)}</b><small>${esc(e.repo)} · ${esc(e.quant||e.file)} · ${esc(e.size_gb??'Unknown')} GB</small><small>${esc(e.fit||'')} ${installed?' · Downloaded':''}</small></div><div class="row">${installed&&!active?'<span class="pill">Downloaded</span>':`<button id="catalog-download-${esc(e.id)}" data-download="${esc(e.id)}" data-active="${!!active}">${active?'Cancel download':'Queue download'}</button>`}<button data-catalogue-remove="${esc(e.id)}" ${active?'disabled':''}>Remove…</button></div></div>`;
 }).join('')||'<p>Your catalogue is empty. Add a variant from the results above.</p>');
}
$('find-form').onsubmit=e=>{e.preventDefault();searchHF();};
$('find-refresh').onclick=()=>searchHF(true);
$('find-clear').onclick=()=>{for(const input of $('find-table').querySelectorAll('thead input'))input.value='';renderFind();};
for(const id of ['find-suitable','find-instruct','find-quants'])$(id).onchange=renderFind;
$('find-table').querySelector('thead').oninput=renderFind;
$('find-table').querySelector('thead').onclick=e=>{const b=e.target.closest('[data-find-sort]');if(!b)return;const key=b.dataset.findSort;findSort=findSort.key!==key?{key,direction:1}:findSort.direction===1?{key,direction:-1}:{key:null,direction:0};renderFind();};
$('find-table').querySelector('tbody').onclick=async e=>{
 const b=e.target.closest('[data-find-add]');if(!b||b.disabled)return;b.disabled=true;
 try{await api('/api/catalogue/add',{id:b.dataset.findAdd});await loadCatalogue();}catch(error){$('find-status').textContent=error.message;b.disabled=false;}
};
$('catalog').onclick=e=>attempt(async()=>{
 const remove=e.target.closest('[data-catalogue-remove]');if(!remove){await downloadClick(e);return;}if(remove.disabled)return;
 const preview=await api('/api/catalogue/removal-preview',{id:remove.dataset.catalogueRemove});
 removingId=remove.dataset.catalogueRemove;$('remove-model-name').textContent=preview.name;
 $('remove-model-files').textContent=preview.paths.join('\n')||'No managed weights or partial downloads found.';
 $('remove-model-weights').checked=false;$('remove-model-confirm').textContent='Remove from catalogue';$('remove-model-error').textContent='';$('remove-model-dialog').showModal();
});
$('remove-model-weights').onchange=()=>{$('remove-model-confirm').textContent=$('remove-model-weights').checked?'Remove and delete weights':'Remove from catalogue';};
$('remove-model-cancel').onclick=()=>$('remove-model-dialog').close();
$('remove-model-confirm').onclick=async()=>{
 $('remove-model-confirm').disabled=true;
 try{await api('/api/catalogue/remove',{id:removingId,delete_weights:$('remove-model-weights').checked});$('remove-model-dialog').close();$('find-query').focus();await scan();await loadCatalogue();await poll();}
 catch(e){$('remove-model-error').textContent=e.message;}
 finally{$('remove-model-confirm').disabled=false;}
};
(async()=>{await poll();await scan();if(['#find','#experiments'].includes(location.hash))await switchView(location.hash.slice(1));slotNote();setInterval(poll,2500);})();
