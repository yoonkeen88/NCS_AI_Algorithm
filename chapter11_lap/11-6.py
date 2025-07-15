n=3
start='-'*(n*n)

def Move(state,pos,player):
    return state[:pos]+player+state[pos+1:]

def switch_player(player):
    return 'X' if player=='O' else 'O'

def print_board(state):
    print('  0123456789012345'[:n+2])
    for i in range(n):
        print(str(i%10)+':'+state[n*i:n*(i+1)])

def get_empty(state):
    if decide_winner(state) in ['O','X','T']: # 승자가 정해지면
        return []
    empty=[]
    for i in range(len(start)):
        if state[i]=='-':
            empty.append(i)
    return empty

def decide_winner(state):
    for (a,b,c) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
        if state[a]==state[b]==state[c]:
            if state[a]=='O': return 'O'
            elif state[a]=='X': return 'X'
    if [i for i in range(n*n) if state[i]=='-']==[]: return 'T' # Tie(비김)
    return 'N' # 아직 승자 정해지지 않음

def minimax(state,player,depth):
    winner=decide_winner(state)
    if winner=='X':
        return 1,None
    elif winner=='O':
        return -1,None
    elif winner=='T':
        return 0,None

    e=get_empty(state)
    if depth%2==0: # 컴퓨터 차례
        vmax,bestpos=-100,None
        for pos in e:
            v,_=minimax(Move(state,pos,player),switch_player(player),depth+1)
            if v>vmax:
                vmax,bestpos=v,pos
        return vmax,bestpos
    else:
        vmin,bestpos=100,None
        for pos in e:
            v,_=minimax(Move(state,pos,player),switch_player(player),depth+1)
            if v<vmin:
                vmin,bestpos=v,pos
        return vmin,bestpos

def tictactoe_play(first_mover):
    state=start
    player=first_mover
    print_board(state)
    while True:
        if player=='X':
            print("컴퓨터 차례입니다.")
            val,pos=minimax(state,player,0)
        elif player=='O':
            x,y=input("사람 차례입니다. (x와 y를 공백 구분하여 입력하세요.)").split()
            pos=int(y)*n+int(x)
            if state[pos]!='-':
                print("둘 수 없는 곳입니다.")
                continue
        state=Move(state,pos,player)
        print_board(state)
        winner=decide_winner(state)
        if winner in ['O','X','T']:
            if winner=='T': print('비겼습니다.')
            else: print(winner,'가 이겼습니다.')
            break
        player=switch_player(player)

tictactoe_play('O')