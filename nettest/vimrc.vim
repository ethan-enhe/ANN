" {{{ ~/.clang-format
" BasedOnStyle: Google
" PointerAlignment: Right
" AccessModifierOffset: -4
" IndentWidth: 4
" MaxEmptyLinesToKeep: 1
" BreakBeforeBraces: Attach
" AllowShortCaseLabelsOnASingleLine: true
" AllowShortFunctionsOnASingleLine: true
" AllowShortIfStatementsOnASingleLine: true
" AllowShortLoopsOnASingleLine: true
" AlignAfterOpenBracket: true
" IndentCaseLabels: true
" ObjCBlockIndentWidth: 4
" ObjCSpaceAfterProperty: true
" ColumnLimit: 120
" AlignTrailingComments: true
" SpaceAfterCStyleCast: false
" AlignOperands: true
" SpacesInSquareBrackets: false
" AlignConsecutiveDeclarations: false
" SpacesInContainerLiterals: false
" BreakConstructorInitializersBeforeComma: true
" AllowAllParametersOfDeclarationOnNextLine: true
" ContinuationIndentWidth: 4
" TabWidth: 4
" SpaceBeforeAssignmentOperators: true
" SpacesBeforeTrailingComments: 1
" }}}
"
"call mkdir(stdpath('config'),'p')
"exe 'edit' stdpath('config').'/init.vim'
" {{{ BASIC
filetype on
filetype plugin on
filetype indent on
try
    call rpcnotify(1, 'Gui', 'Option', 'Tabline', 0)
    call rpcnotify(1, 'Gui', 'Option', 'Popupmenu', 0)
catch
endtry
let mapleader=","
set guifont=Consolas:h14
set guioptions=

set backspace=indent,eol,start
set whichwrap+=<,>,h,l
set winaltkeys=no

let g:iswindows = 0
if(has("win32") || has("win64") || has("win95") || has("win16"))
    let g:iswindows = 1
endif
let g:usecoc = 0
let g:usefont = 1

set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
set ffs=unix,dos,mac
noremap <Leader>m mmHmt:%s/<C-V><cr>//ge<cr>'tzt'm
noremap <leader>rt <cmd>ret<cr>

set number
set relativenumber
map <silent> <leader>rl :set relativenumber!<cr>
set mouse=a
set clipboard+=unnamedplus

" TextEdit might fail if hidden is not set.
set hidden

" Auto
set autochdir
set autowrite
set autoread
" au FocusGained,BufEnter * checktime

map <leader>c <cmd>cd %:p:h<cr>:pwd<cr>
nmap <leader>w <cmd>w!<cr>
"}}}
"{{{ BACKUP
set nobackup
set nowritebackup
set noswapfile
set udf
"}}}
"{{{ INTERFACE
set completeopt=menu,menuone,noselect
set ruler
set showcmd
set cmdheight=2
set laststatus=2
set scrolloff=7
let $LANG='en'
set langmenu=en
set wildmenu
" Ignore compiled files
set wildignore=*.o,*~,*.pyc
if has("win16") || has("win32")
    set wildignore+=.git\*,.hg\*,.svn\*
else
    set wildignore+=*/.git/*,*/.hg/*,*/.svn/*,*/.DS_Store
endif
set hlsearch
set incsearch
set ignorecase
set smartcase
set diffopt+=vertical
set lazyredraw
set magic
" set showmatch
" set mat=2

set updatetime=300
set noerrorbells
set novisualbell
set tm=500
if has("nvim-0.5.0") || has("patch-8.1.1564")
    " Recently vim can merge signcolumn and number column into one
    set signcolumn=number
else
    set signcolumn=yes
endif
"}}}
"{{{ TEXT
set ts=4
set sw=4
set expandtab
set foldenable
set foldmethod=marker
" set foldlevel=99
" Indent+Cursor
set smartindent
set autoindent
let g:AutoPairsMapSpace=0
let g:AutoPairsMultilineClose=0
"}}}
"{{{ VISUAL
vnoremap <silent> * :<C-u>call VisualSelection('', '')<cr>/<C-R>=@/<cr><cr>
vnoremap <silent> # :<C-u>call VisualSelection('', '')<cr>?<C-R>=@/<cr><cr>
"}}}
"{{{ MOVING
map <C-h> <C-W>h
map <C-j> <C-W>j
map <C-k> <C-W>k
map <C-l> <C-W>l
" map <space> /
" map <C-space> ?
" Close the current buffer
map <leader>bd <cmd>Bclose<cr>

" Close all the buffers
map <leader>ba <cmd>bufdo bd<cr>

map <leader>l <cmd>bnext<cr>
map <leader>h <cmd>bprevious<cr>

" Useful mappings for managing tabs
map <leader>tn <cmd>tabnew<cr>
map <leader>to <cmd>tabonly<cr>
map <leader>tc <cmd>tabclose<cr>
map <leader>tm :tabmove
map <leader>t<leader> :tabnext
" Let 'tl' toggle between this and the last accessed tab
let g:lasttab = 1
nmap <Leader>tl <cmd>exe "tabn ".g:lasttab<cr>
au TabLeave * let g:lasttab = tabpagenr()
" Opens a new tab with the current buffer's path
" Super useful when editing files in the same directory
map <leader>te <cmd>tabedit <C-r>=expand("%:p:h")<cr>/
" Specify the behavior when switching between buffers
try
    set switchbuf=useopen,usetab,newtab
    set stal=2
catch
endtry
command! Bclose call <SID>BufcloseCloseIt()
function! <SID>BufcloseCloseIt()
    let l:currentBufNum = bufnr("%")
    let l:alternateBufNum = bufnr("#")

    if buflisted(l:alternateBufNum)
        buffer #
    else
        bnext
    endif

    if bufnr("%") == l:currentBufNum
        new
    endif

    if buflisted(l:currentBufNum)
        execute("bdelete! ".l:currentBufNum)
    endif
endfunction
" Return to last edit position when opening files (You want this!)
au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
" let g:smoothie_experimental_mappings=1
"}}}
" {{{ MAP
" if !g:usecoc
"   inoremap ' ''<LEFT>
"   inoremap " ""<LEFT>
"   inoremap ( ()<LEFT>
"   inoremap [ []<LEFT>
"   inoremap { {}<LEFT>
" endif
map 0 ^
inoremap jj <esc>
tnoremap jj <c-\><c-n>
tnoremap <esc> <c-\><c-n>
" Move a line of text using ALT+[jk] or Command+[jk] on mac
nmap <M-j> mz:m+<cr>`z
nmap <M-k> mz:m-2<cr>`z
vmap <M-j> <cmd>m'>+<cr>`<my`>mzgv`yo`z
vmap <M-k> <cmd>m'<-2<cr>`>my`<mzgv`yo`z
" Quickly open a buffer for scribble
map <leader>x <cmd>e ~/buffer<cr>

" Quickly open a markdown buffer for scribble
" map <leader>x <cmd>e ~/buffer.md<cr>

" Toggle paste mode on and off
map <leader>pp <cmd>setlocal paste!<cr>

function! CmdLine(str)
    call feedkeys(":" . a:str)
endfunction

function! VisualSelection(direction, extra_filter) range
    let l:saved_reg = @"
    execute "normal! vgvy"

    let l:pattern = escape(@", "\\/.*'$^~[]")
    let l:pattern = substitute(l:pattern, "\n$", "", "")

    if a:direction == 'gv'
        call CmdLine("Ack '" . l:pattern . "' " )
    elseif a:direction == 'replace'
        call CmdLine("%s" . '/'. l:pattern . '/')
    endif

    let @/ = l:pattern
    let @" = l:saved_reg
endfunction

"map <F10> <cmd>NERDTreeToggle<cr>
"}}}
"{{{ FILE SETTING
map <F10> <cmd>MarkdownPreviewToggle<cr>
map <leader>tt <cmd>0 r ~/code/template/other/cf.cpp<cr>
map <leader>tr :r ~/code/template/
map <C-c> :Commentary<cr>
" autocmd FileType cpp setlocal commentstring=//%s cindent
autocmd FileType cpp setlocal cindent
map <F8> <cmd>call RunCode()<cr>
map <leader>ts <cmd>call RuninFloat('cf test')<cr>
map <leader>sm <cmd>call RuninFloat('cf submit')<cr>
map <F9> <cmd>call CompileCode('-lm -O2 -DLOCAL -Wall -fsanitize=address,undefined')<cr>
map <leader><F9> <cmd>call CompileCode('-lm -Wall -O3  -fopenmp -lpthread')<cr>
let s:res=""
function! s:OnEvent(job_id, data, event) dict
    if a:event == 'exit'
        echo s:res."finished."
        let s:res=""
    else
        let s:res =s:res.join(a:data,"\n")
    endif
endfunction
let s:callbacks = {
            \ 'on_stdout': function('s:OnEvent'),
            \ 'on_stderr': function('s:OnEvent'),
            \ 'on_exit': function('s:OnEvent')
            \ }

let g:pauser=g:iswindows?'pause':'bash -c "read -p Press\ enter\ to\ continue..."'
func! CompileCode(copt)
    exec "w"
    echo "compiling... ".a:copt
    if &filetype=='cpp'
        let cpl=jobstart((g:iswindows?"g++ -Wl,-stack=536870912":"g++")."\ ".a:copt."\ ".expand("%")."\ -o\ ".expand("%:t:r"),s:callbacks)
    else
        let cpl=jobstart("gcc"."\ ".expand("%")."\ -lm -o\ ".expand("%:t:r"),s:callbacks)
    endif
endfunction
func! RunCode()
    exec "w"
    let s:pre=has("nvim")?"bel 10sp term://":"!"
    if &filetype == 'cpp' || &filetype=='c'
        let s:suf=g:iswindows?expand('%:t:r').'.exe':'\time -f "\n----\n\%Mkb \%Us" ./'.expand('%:t:r')
    elseif &filetype == 'python'
        let s:suf='python3 '.expand('%')
    elseif &filetype == 'lua'
        let s:suf='lua '.expand('%')
    endif

    call RuninFloat(s:suf)
    " exec 'FloatermSend cd '.expand('%:p:h').' && '.s:suf
    " exec 'FloatermShow'
    " call RunCmdinCwd(s:suf)
    " exec 'FloatermSend cd '.expand('%:p:h').' && '.s:suf
    " exec 'FloatermShow'
endfunction
fun! RuninFloat(cmd)
    exec 'FloatermNew --autoclose=2 '.a:cmd.' && '.g:pauser
endfun
fun! RuninTerminal(cmd)
    exec 'FloatermSend cd '.expand('%:p:h').' && '.a:cmd
    exec 'FloatermShow'
endfun
" Delete trailing white space on save, useful for some filetypes ;)
fun! CleanExtraSpaces()
    let save_cursor = getpos(".")
    let old_query = getreg('/')
    silent! %s/\s\+$//e
    call setpos('.', save_cursor)
    call setreg('/', old_query)
endfun

if has("autocmd")
    autocmd BufWritePre *.md,*.cpp,*.txt,*.js,*.py,*.wiki,*.sh,*.coffee :call CleanExtraSpaces()
endif
"}}}
" {{{ PLUG
if empty(glob(stdpath('config')."/autoload/"))
    exec "!curl -fLo ".stdpath('config')."/autoload/plug.vim --create-dirs https://ethan_enhe.coding.net/p/code/d/code/git/raw/master/template/vim/plug.vim"
endif

" let g:plug_url_format='https://github.com.cnpmjs.org/%s'
let g:plug_url_format='https://hub.fastgit.xyz/%s'
" let g:plug_url_format='https://gitclone.com/github.com/%s'
call plug#begin()
Plug 'luochen1990/rainbow'
Plug 'morhetz/gruvbox'
Plug 'sainnhe/gruvbox-material'
Plug 'sainnhe/edge'
Plug 'joshdick/onedark.vim'
Plug 'mhartington/oceanic-next'
" Plug 'psliwka/vim-smoothie'
Plug 'jiangmiao/auto-pairs'
Plug 'Yggdroot/indentLine'


Plug 'ryanoasis/vim-devicons'
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']}

Plug 'voldikss/vim-floaterm'


" Plug 'ZSaberLv0/ZFVimJob'
" Plug 'ZSaberLv0/ZFVimGitUtil'
" Plug 'ethan-enhe/ZFVimIM'
" Plug 'qdzhang/ZFVimIM_xiaohe'
" Plug 'ZSaberLv0/ZFVimIM_openapi' " 百度云输入法
" Plug 'ethan-enhe/ZFVimIM_pinyin_base' " 你的词库, 确保有 push 权限
if g:usecoc
    Plug 'ethan-enhe/vim-snippets/'
    Plug 'neoclide/coc.nvim', {'branch': 'release'}
    Plug 'jackguo380/vim-lsp-cxx-highlight'
else
    " Plug 'github/copilot.vim'
    " Plug 'w0rp/ale'
    " Plug 'neovim/nvim-lspconfig'
    " Plug 'hrsh7th/cmp-nvim-lsp'
    " Plug 'hrsh7th/cmp-buffer'
    " Plug 'hrsh7th/cmp-path'
    " Plug 'hrsh7th/cmp-cmdline'
    " Plug 'hrsh7th/nvim-cmp'

    " Plug 'SirVer/ultisnips'
    " Plug 'quangnguyen30192/cmp-nvim-ultisnips'
endif
call plug#end()
"}}}
"{{{
let g:ZFVimIM_pinyin_gitUserEmail='18201521626@139.com'
let g:ZFVimIM_pinyin_gitUserName='ethan-enhe'
let g:ZFVimIM_pinyin_gitUserToken='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'
"}}}
" {{{ COLOR
if $COLORTERM == 'gnome-terminal'
    set t_Co=256
endif
let g:rainbow_active = 1
let g:gruvbox_sign_column='bg0'
let g:gruvbox_color_column='bg0'
let g:gruvbox_number_column='bg0'
let g:gruvbox_vert_split='bg0'
let g:onedark_terminal_italics=1

" let g:edge_style = 'neon'
let g:edge_transparent_background = 1
" let g:edge_diagnostic_text_highlight = 1

colorscheme gruvbox-material
" autocmd vimenter * hi Normal guibg=NONE ctermbg=NONE
let g:indent_guides_enable_on_vim_startup = 1
let g:indentLine_char = '|'
" let g:indentLine_char_list = ['|', '¦', '┆', '┊']
" let g:indentLine_leadingSpaceEnabled=1
let g:indentLine_bufTypeExclude = ['help', 'terminal']
let g:indentLine_bufNameExclude = ['_.*','[coc-explorer]-*', 'NERD_tree.*']


set background=dark
set list lcs=tab:«·»,nbsp:+,trail:·,extends:→,precedes:←
set termguicolors
set noshowmode
syntax enable
map <silent> <leader><cr> :noh<cr>

" let g:airline_theme='onedark'
if g:usefont
    let g:airline_powerline_fonts = 1
    let g:airline_left_sep = "\ue0bc"
    let g:airline_left_alt_sep = "\ue0bd"
    let g:airline_right_sep = "\ue0be"
    let g:airline_right_alt_sep = "\ue0bf"
endif
let g:airline#extensions#tabline#enabled = 1 "tabline中当前buffer两端的分隔字符
let g:airline#extensions#whitespace#enabled = 0
" let g:airline#extensions#tabline#buffer_nr_show=1
"}}}
"{{{ FLOATERM
let g:floaterm_keymap_toggle = '<f12>'
let g:floaterm_width=0.8
let g:floaterm_height=0.8
" let g:floaterm_autoinsert=0
" let g:floaterm_position='bottomright'
let g:floaterm_opener='vsplit'
hi FloatermBorder guibg=none
hi FloatermNC guifg=gray
au VimEnter * FloatermNew --silent
"}}}
if g:usecoc
    "{{{ COC_CONFIG

    let g:buffersource=1
    command! CocBufSourceToggle call ToggleBufferSource()
    function ToggleBufferSource()
        let g:buffersource=!g:buffersource
        call coc#config("coc.source.around.enable", g:buffersource)
        call coc#config("coc.source.buffer.enable", g:buffersource)
        echo 'ToggleBufferSource '.g:buffersource
    endfunction
    " call coc#config("coc.source.around.enable", v:false)
    call coc#config("diagnostic.virtualText", v:true)
    call coc#config("diagnostic.virtualTextCurrentLineOnly", v:false)
    call coc#config("diagnostic.virtualTextPrefix", " ❯❯❯ ")
    call coc#config("suggest.labelMaxLength", 50)
    " call coc#config("suggest.enablePreview", v:true)
    " call coc#config("semanticTokens.enable",v:true)
    " call coc#config("coc.semanticHighlighting", v:true)
    " call coc#config("coc.preferences.semanticTokensHighlights", v:false)
    
    call coc#config("diagnostic.errorSign", "✘")
    call coc#config("diagnostic.warningSign", "")
    call coc#config("diagnostic.infoSign", "")
    call coc#config("diagnostic.hintSign", "•")
    call coc#config("suggest.completionItemKindLabels",{
                \       "keyword": "\uf1de",
                \       "variable": "\ue79b",
                \       "value": "\uf89f",
                \       "operator": "\u03a8",
                \       "constructor": "\uf0ad",
                \       "function": "\u0192",
                \       "reference": "\ufa46",
                \       "constant": "\uf8fe",
                \       "method": "\uf09a",
                \       "struct": "\ufb44",
                \       "class": "\uf0e8",
                \       "interface": "\uf417",
                \       "text": "\ue612",
                \       "enum": "\uf435",
                \       "enumMember": "\uf02b",
                \       "module": "\uf40d",
                \       "color": "\ue22b",
                \       "property": "\ue624",
                \       "field": "\uf9be",
                \       "unit": "\uf475",
                \       "event": "\ufacd",
                \       "file": "\uf723",
                \       "folder": "\uf114",
                \       "snippet": "\ue60b",
                \       "typeParameter": "\uf728",
                \       "default": "\uf29c"
                \   })
    "}}}
    "{{{ COC
    " let g:coc_global_extensions = ['coc-clangd','coc-markdownlint','coc-vimlsp', 'coc-json','coc-snippets','coc-lists','coc-explorer','coc-floaterm']
    let g:coc_global_extensions = ['coc-clangd','coc-markdownlint','coc-vimlsp', 'coc-json','coc-snippets','coc-lists','coc-floaterm']
    autocmd FileType * let b:coc_pairs_disabled = ['<']
    " Having longer updatetime (default is 4000 ms = 4 s) leads to noticeable
    " delays and poor user experience.
    set updatetime=300

    " Don't pass messages to |ins-completion-menu|.
    set shortmess+=c

    " Always show the signcolumn, otherwise it would shift the text each time
    " diagnostics appear/become resolved.


    " Use tab for trigger completion with characters ahead and navigate.
    " NOTE: Use command ':verbose imap <tab>' to make sure tab is not mapped by
    " other plugin before putting this into your config.
    inoremap <silent><expr> <TAB>
                \ coc#pum#visible() ? coc#pum#next(1) :
                \ CheckBackspace() ? "\<Tab>" :
                \ coc#refresh()

    inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"


    function! CheckBackspace() abort
        let col = col('.') - 1
        return !col || getline('.')[col - 1]  =~# '\s'
    endfunction

    " Use <c-space> to trigger completion.
    if has('nvim')
        inoremap <silent><expr> <c-space> coc#refresh()
    else
        inoremap <silent><expr> <c-@> coc#refresh()
    endif

    " inoremap <silent><expr> <cr> pumvisible() ? "\<C-y>" : "\<cr>"
    inoremap <silent><expr> <CR> coc#pum#visible() ? coc#pum#confirm()
                                  \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"
    " let g:coc_default_semantic_highlight_groups = 1
    " Use `[g` and `]g` to navigate diagnostics
    " Use `:CocDiagnostics` to get all diagnostics of current buffer in location list.
    nmap <silent> [g <Plug>(coc-diagnostic-prev)
    nmap <silent> ]g <Plug>(coc-diagnostic-next)

    " GoTo code navigation.
    nmap <silent> gd <Plug>(coc-definition)
    nmap <silent> gy <Plug>(coc-type-definition)
    nmap <silent> gi <Plug>(coc-implementation)
    nmap <silent> gr <Plug>(coc-references)

    map <leader>rs <cmd>CocRestart<cr>
    map <leader>bs <cmd>CocBufSourceToggle<cr>

    " Use K to show documentation in preview window.
    nnoremap <silent> K :call ShowDocumentation()<CR>
    nnoremap <silent> H :h <c-r>=expand('<cword>')<cr><cr>

    function! ShowDocumentation()
        if CocAction('hasProvider', 'hover')
            call CocActionAsync('doHover')
        else
            call feedkeys('K', 'in')
        endif
    endfunction

    " Highlight the symbol and its references when holding the cursor.
    autocmd CursorHold * silent call CocActionAsync('highlight')

    " Symbol renaming.
    nmap <leader>rn <Plug>(coc-rename)
    nmap <leader>rf <Plug>(coc-refactor)


    " Formatting selected code.
    " xmap <leader>f    <Plug>(coc-format-selected)
    " nmap <leader>f    <Plug>(coc-format-selected)
    map <leader>f   <cmd>Format<cr>


    augroup mygroup
        autocmd!
        " Setup formatexpr specified filetype(s).
        autocmd FileType typescript,json setl formatexpr=CocAction('formatSelected')
        " Update signature help on jump placeholder.
        autocmd User CocJumpPlaceholder call CocActionAsync('showSignatureHelp')
    augroup end

    " Applying codeAction to the selected region.
    " Example: `<leader>aap` for current paragraph
    xmap <leader>a  <Plug>(coc-codeaction-selected)
    nmap <leader>a  <Plug>(coc-codeaction-selected)

    " Remap keys for applying codeAction to the current buffer.
    nmap <leader>ac <Plug>(coc-codeaction)
    " Apply AutoFix to problem on the current line.
    nmap <leader>qf <Plug>(coc-fix-current)

    " Map function and class text objects
    " NOTE: Requires 'textDocument.documentSymbol' support from the language server.
    xmap if <Plug>(coc-funcobj-i)
    omap if <Plug>(coc-funcobj-i)
    xmap af <Plug>(coc-funcobj-a)
    omap af <Plug>(coc-funcobj-a)
    xmap ic <Plug>(coc-classobj-i)
    omap ic <Plug>(coc-classobj-i)
    xmap ac <Plug>(coc-classobj-a)
    omap ac <Plug>(coc-classobj-a)

    " Remap <C-f> and <C-b> for scroll float windows/popups.
    if has('nvim-0.4.0') || has('patch-8.2.0750')
        nmap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? coc#float#scroll(1) : "\<C-f>"
        nmap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? coc#float#scroll(0) : "\<C-b>"
        imap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? "\<c-r>=coc#float#scroll(1)\<cr>" : "\<Right>"
        imap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? "\<c-r>=coc#float#scroll(0)\<cr>" : "\<Left>"
        vmap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? coc#float#scroll(1) : "\<C-f>"
        vmap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? coc#float#scroll(0) : "\<C-b>"
    endif

    " Use CTRL-S for selections ranges.
    " Requires 'textDocument/selectionRange' support of language server.
    nmap <silent> <C-s> <Plug>(coc-range-select)
    xmap <silent> <C-s> <Plug>(coc-range-select)

    " Add `:Format` command to format current buffer.
    command! -nargs=0 Format :call CocAction('format')

    " Add `:Fold` command to fold current buffer.
    command! -nargs=? Fold :call     CocAction('fold', <f-args>)

    " Add `:OR` command for organize imports of the current buffer.
    command! -nargs=0 OR     :call   CocAction('runCommand', 'editor.action.organizeImport')
    " Add (Neo)Vim's native statusline support.
    " NOTE: Please see `:h coc-status` for integrations with external plugins that
    " provide custom statusline: lightline.vim, vim-airline.
    " set statusline^=%{coc#status()}%{get(b:,'coc_current_function','')}

    " Mappings for CoCList
    nnoremap <silent><nowait> <space>l  :<C-u>CocList<cr>
    " Show all diagnostics.
    nnoremap <silent><nowait> <space>a  :<C-u>CocList diagnostics<cr>
    " Manage extensions.
    " nnoremap <silent><nowait> <space>e    :<C-u>CocList extensions<cr>
    " Show commands.
    nnoremap <silent><nowait> <space>c  :<C-u>CocList commands<cr>
    nnoremap <silent><nowait> <space>f  :<C-u>CocList --auto-preview floaterm<cr>
    " Find symbol of current document.
    nnoremap <silent><nowait> <space>o  :<C-u>CocList outline<cr>
    nnoremap <silent><nowait> <space>g  :<C-u>CocList grep<cr>
    " Search workspace symbols.
    nnoremap <silent><nowait> <space>s  :<C-u>CocList -I symbols<cr>
    " Do default action for next item.
    nnoremap <silent><nowait> <space>j  :<C-u>CocNext<cr>
    " Do default action for previous item.
    nnoremap <silent><nowait> <space>k  :<C-u>CocPrev<cr>
    " Resume latest coc list.
    nnoremap <silent><nowait> <space>p  :<C-u>CocListResume<cr>
    "}}}
    "{{{ COC-SNIP
    " Use <C-l> for trigger snippet expand.
    imap <C-l> <Plug>(coc-snippets-expand)
    " Use <C-j> for select text for visual placeholder of snippet.
    vmap <C-j> <Plug>(coc-snippets-select)
    " Use <C-j> for jump to next placeholder, it's default of coc.nvim
    let g:coc_snippet_next = '<c-j>'
    " Use <C-k> for jump to previous placeholder, it's default of coc.nvim
    let g:coc_snippet_prev = '<c-k>'
    " Use <C-j> for both expand and jump (make expand higher priority.)
    imap <C-j> <Plug>(coc-snippets-expand-jump)
    " Use <leader>x for convert visual selected code to snippet
    xmap <leader>x  <Plug>(coc-convert-snippet)
    "}}}
    " {{{ COC-EXP
    " nnoremap <silent><nowait> <space>e  <Cmd>CocCommand explorer<cr>
    " " nmap <space>el <Cmd>CocList explPresets<cr>
    " augroup vime_coc_explorer_group
    "     autocmd!
    "     " autocmd WinEnter * if &filetype == 'coc-explorer' && winnr('$') == 1 | q | endif
    "     autocmd TabLeave * if &filetype == 'coc-explorer' | wincmd w | endif
    " augroup END
    " let g:coc_explorer_global_presets = {
    " \   '.vim': {
    " \     'root-uri': '~/.vim',
    " \   },
    " \   'cocConfig': {
    " \      'root-uri': '~/.config/coc',
    " \   },
    " \   'tab': {
    " \     'position': 'tab',
    " \     'quit-on-open': v:true,
    " \   },
    " \   'tab:$': {
    " \     'position': 'tab:$',
    " \     'quit-on-open': v:true,
    " \   },
    " \   'floating': {
    " \     'position': 'floating',
    " \     'open-action-strategy': 'sourceWindow',
    " \   },
    " \   'floatingTop': {
    " \     'position': 'floating',
    " \     'floating-position': 'center-top',
    " \     'open-action-strategy': 'sourceWindow',
    " \   },
    " \   'floatingLeftside': {
    " \     'position': 'floating',
    " \     'floating-position': 'left-center',
    " \     'floating-width': 50,
    " \     'open-action-strategy': 'sourceWindow',
    " \   },
    " \   'floatingRightside': {
    " \     'position': 'floating',
    " \     'floating-position': 'right-center',
    " \     'floating-width': 50,
    " \     'open-action-strategy': 'sourceWindow',
    " \   },
    " \   'simplify': {
    " \     'file-child-template': '[selection | clip | 1] [indent][icon | 1] [filename omitCenter 1]'
    " \   },
    " \   'buffer': {
    " \     'sources': [{'name': 'buffer', 'expand': v:true}]
    " \   },
    " \ }
    " call coc#config("explorer.icon.enableNerdfont", g:usefont)
    " call coc#config("explorer.contentWidthType", "win-width")
    " call coc#config("explorer.bookmark.child.template", "[selection | 1] [filename] [position] - [annotation]")
    " call coc#config("explorer.file.column.icon.modified", "•")
    " call coc#config("explorer.file.column.icon.deleted", "✗")
    " call coc#config("explorer.file.column.icon.untracked", "★")
    " call coc#config("explorer.file.column.icon.renamed", "➜")
    " call coc#config("explorer.file.column.icon.unmerged", "")
    " call coc#config("explorer.file.column.icon.ignored", "ⁱ")
    " call coc#config("explorer.keyMappings.global", {
    "             \ 's': v:false,
    "             \ 't': v:false,
    "             \ 'E': v:false,
    "             \ 'e': v:false,
    "             \ 'zh': v:false,
    "             \ 'g.': v:false,
    "             \ 'p': v:false,
    "             \ })
    " call coc#config("explorer.keyMappings.global", {
    "             \ 'k': 'nodePrev',
    "             \ 'j': 'nodeNext',
    "             \ 'h': ["wait", 'collapse'],
    "             \ 'l': ["wait", 'expandable?', 'expand', 'open'],
    "             \ 'L': ["wait", 'expand:recursive'],
    "             \ 'H': ["wait", 'collapse:recursive'],
    "             \ 'K': ["wait", 'expandablePrev'],
    "             \ 'J': ["wait", 'expandableNext'],
    "             \ 'o': ["wait", 'expanded?', 'collapse', 'expand'],
    "             \ '<cr>': ["wait", 'expandable?', 'cd', 'open'],
    "             \ '<bs>': ["wait", 'gotoParent'],
    "             \ 'r': 'refresh',
    "             \
    "             \ 's': ["wait", 'toggleSelection', 'normal:j'],
    "             \ 'S': ["wait", 'toggleSelection', 'normal:k'],
    "             \ '*': ["wait", 'toggleSelection'],
    "             \ 'gs': ["wait", "reveal:select"],
    "             \ '<dot>': 'toggleHidden',
    "             \
    "             \ '<c-s>': 'open:split',
    "             \ '<c-v>': 'open:vsplit',
    "             \ '<c-t>': 'open:tab',
    "             \
    "             \ 'dd': 'cutFile',
    "             \ 'Y': 'copyFile',
    "             \ 'D': 'delete',
    "             \ 'P': 'pasteFile',
    "             \ 'R': 'rename',
    "             \ 'N': 'addFile',
    "             \ 'yp': 'copyFilepath',
    "             \ 'yn': 'copyFilename',
    "             \
    "             \ 'pl': 'previewOnHover:toggle:labeling',
    "             \ 'pc': 'previewOnHover:toggle:content',
    "             \
    "             \ '<M-x>': 'systemExecute',
    "             \ 'f': 'search',
    "             \ 'F': 'searchRecursive',
    "             \
    "             \ '<tab>': 'actionMenu',
    "             \ '?': 'help',
    "             \ 'q': 'quit',
    "             \ '<esc>': 'esc',
    "             \
    "             \ 'gf': 'gotoSource:file',
    "             \ 'gb': 'gotoSource:buffer',
    "             \ '[[': ["wait", 'indentPrev'],
    "             \ ']]': ["wait", 'indentNext'],
    "             \
    "             \ '<M-k>': ["wait", 'markPrev:diagnosticError'],
    "             \ '<M-j>': ["wait", 'markNext:diagnosticError'],
    "             \
    "             \ '<leader>gk': ["wait", 'markPrev:git'],
    "             \ '<leader>gj': ["wait", 'markNext:git'],
    "             \ '<leader>gh': 'gitStage',
    "             \ '<leader>gu': 'gitUnstage'
    "             \ })
    " }}}
else
endif
